from typing import Tuple, Union
from pathlib import Path as pth
import json
import tempfile
from shutil import rmtree, copy
from typing import Union, Optional, List
import os
from random import random, choice

import fire
import numpy as np
from PIL import Image

from prepare_data import (
    get_hdris,
    get_materials,
    DownloadError
)
from blender_render_samples_3d import run_blender_command
from postprocess import postprocess_samples
from Blender_3D_document_rendering_pipeline.src import config

class SampleInfo:
    def __init__(
        self,
        image_path: pth,
        config: config.Config,
        font_size: int  = 0,
        text: str = "",  # 保留但不使用
        font_path: Optional[str] = None,  # 可以为None
        font_color: np.ndarray = np.array([0, 0, 0]),  # 默认值
        text_rotation_angle: int = 0,  # 默认值
        output_image_resolution: Tuple[int, int] = (512, 512),
        config_path: Optional[pth] = None,
        output_image_path: Optional[pth] = None,
        output_coordinates_path: Optional[pth] = None,
        compression_level: int = 9,
    ):
        self.config = config
        self.text = text  # 不使用，但保留兼容性
        self.config_path = config_path
        self.image_path = image_path
        self.font_size = font_size
        self.font_path = font_path  # 不使用，但保留兼容性
        self.font_color = font_color  # 不使用，但保留兼容性
        self.text_rotation_angle = text_rotation_angle  # 不使用，但保留兼容性
        self.output_image_resolution = output_image_resolution
        self.output_image_path = output_image_path
        self.output_coordinates_path = output_coordinates_path if output_coordinates_path else (output_image_path.parent / "coordinates0001.png" if output_image_path else None)
        self.compression_level = compression_level


def prepare_resources():
    # 资源存放目录
    download_path = pth("assets/").resolve()
    
    if not download_path.is_dir():
        raise FileNotFoundError(f"资源目录 {download_path} 不存在，请确保已手动下载并解压资源。")

    # 直接使用手动下载的本地资源路径
    hdris_path = download_path / "hdris"
    materials_path = download_path / "materials"

    # 确保资源目录存在
    for path, name in [(hdris_path, "HDRI"), (materials_path, "材质")]:
        if not path.is_dir():
            raise FileNotFoundError(f"错误：{name} 资源目录 {path} 不存在，请检查路径是否正确。")

    # 加载资源
    hdris_iter = get_hdris(hdris_path)
    materials_iter = get_materials(materials_path)

    return hdris_iter, materials_iter

def assign_material_to_conf(material, conf):
    material = {k: str(material[k].resolve()) for k in material}

    conf.ground.albedo_tex = material["albedo"]
    conf.ground.roughness_tex = material["roughness"]
    conf.ground.displacement_tex = material["displacement"]

def write_config(
    img: np.ndarray,
    device: str,
    root_dir: pth,
    output_dir: pth,
    hdri_path,
    material,
    output_image_resolution,
    compression_level: int,
    image_path,
    config_path: pth,
) -> config.Config:
    resolution = np.array(img.shape[:2], np.float32)
    paper_size = resolution[::-1] / np.mean(resolution) * 25

    conf = config.Config(device, project_root=root_dir)

    conf.hdri.texture_path = str(pth(hdri_path).resolve())

    assign_material_to_conf(material, conf)

    conf.render.output_dir = str(output_dir)
    conf.render.resolution = output_image_resolution
    conf.render.cycles_samples = 2
    conf.render.compression_ratio = round(compression_level / 9 * 100)
    conf.paper.document_image_path = str(image_path.resolve(True))
    conf.paper.size = paper_size.tolist()
    conf.ground.visible = random() < 0.6

    config.write_config(config_path, conf)

    return conf

def create_sample_from_image(
    image_path: Union[str, pth],
    output_dir: pth,
    config_dir: pth,
    image_dir: pth,
    root_dir:pth,
    random_hdri_iter,
    random_material_iter,
    resolution=(512, 512),
    compression_level=9,
    device='cpu'
) -> SampleInfo:
    """
    从已有图片创建Sample对象以供Blender处理
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        config_dir: 配置文件目录
        image_dir: 临时图片目录
        random_hdri_iter: HDRI迭代器
        random_material_iter: 材质迭代器
        resolution: 输出分辨率
        compression_level: 压缩级别
        
    Returns:
        Sample对象
    """
    image_path = pth(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"输入图片 {image_path} 不存在")
    
    # 为样本创建一个唯一ID和对应的目录
    sample_id = f"sample_{image_path.stem}"
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制输入图片到临时目录
    temp_image_path = image_dir / f"{sample_id}.png"
    
    # 读取输入图像，调整大小并保存
    input_image = Image.open(image_path)
    input_image.save(temp_image_path, compression=compression_level)
    img_array = np.array(input_image)
    # 创建配置文件
    config_path = config_dir / f"{sample_id}.json"
    
    # 随机选择资源 - 只需要HDRI和材质
    hdri_path = next(random_hdri_iter)
    material = next(random_material_iter)

    conf = write_config(
        img_array,
        device,
        root_dir,
        output_dir,
        hdri_path,
        material,
        resolution,
        compression_level,
        image_path,
        config_path,
    )

    # 保存配置
    # with open(config_path, "w") as f:
    #     json.dump(config, f, indent=4)
    
    # 创建Sample对象 - 简化版本
    sample = SampleInfo(
        temp_image_path,
        conf,
    )
    
    return sample


def process_image_to_3d(
    input_image_path: str,
    blender_path: str,
    output_dir: str,
    device: str,
    resolution_x: int = 512,
    resolution_y: int = 512,
    compression_level: int = 9,
):
    """
    将一张输入图片直接进行3D渲染
    
    Args:
        input_image_path (str): 输入图片路径
        blender_path (str): Blender可执行文件路径
        output_dir (str): 输出目录
        device (str): 渲染设备 ('cpu', 'cuda' or 'optix')
        resolution_x (int): 输出图像X分辨率
        resolution_y (int): 输出图像Y分辨率
        compression_level (int): PNG压缩级别(0-9)
    """
    print("开始加载Blender...")

    device = device.upper()
    input_image_path = pth(input_image_path)
    blender_path = pth(blender_path)
    output_dir = pth(output_dir)

    if device not in ["CPU", "CUDA", "OPTIX"]:
        raise ValueError(f"无效的设备: {device}")

    if not input_image_path.is_file():
        raise ValueError(f"输入图片 {input_image_path} 不是有效文件")

    if not blender_path.is_file():
        raise ValueError(f"Blender路径 {blender_path} 不是有效文件")

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    blender_path = blender_path.resolve()
    output_dir = output_dir.resolve()

    resolution = resolution_x, resolution_y

    print("开始准备资源...")
    # 移除字体迭代器
    hdris, materials = prepare_resources()

    root_dir = pth.cwd() / "Blender_3D_document_rendering_pipeline"

    temp_dir = pth(tempfile.mkdtemp())
    print(f"临时文件保存到: {temp_dir}")

    config_dir = temp_dir / "configs"
    image_dir = temp_dir / "images"
    config_dir.mkdir()
    image_dir.mkdir()

    print(f"处理输入图片: {input_image_path}")
    sample = create_sample_from_image(
        image_path=input_image_path,
        output_dir=output_dir,
        config_dir=config_dir,
        root_dir=root_dir,
        image_dir=image_dir,
        random_hdri_iter=hdris,
        random_material_iter=materials,
        resolution=resolution,
        compression_level=compression_level
    )
    
    generated_samples = [sample]

    print("使用Blender进行3D渲染...")
    run_blender_command(blender_path, config_dir, output_dir, device)

    print("进行后处理...")
    postprocess_samples(generated_samples)

    # 清理临时文件
    print("清理临时文件...")
    rmtree(temp_dir)
    
    print(f"处理完成！输出目录: {output_dir}")


def main(
    input_image_path: str,
    blender_path: str,
    output_dir: str,
    device: str,
    resolution_x: int = 512,
    resolution_y: int = 512,
    compression_level: int = 9,
):
    """
    将一张输入图片直接进行3D渲染
    
    Args:
        input_image_path (str): 输入图片路径
        blender_path (str): Blender可执行文件路径
        output_dir (str): 输出目录
        device (str): 渲染设备 ('cpu', 'cuda' or 'optix')
        resolution_x (int): 输出图像X分辨率
        resolution_y (int): 输出图像Y分辨率
        compression_level (int): PNG压缩级别(0-9)
    """
    process_image_to_3d(
        input_image_path=input_image_path,
        blender_path=blender_path,
        output_dir=output_dir,
        device=device,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        compression_level=compression_level
    )


def batch_process(
    input_dir: str,
    blender_path: str,
    output_dir: str,
    device: str,
    resolution_x: int = 512,
    resolution_y: int = 512,
    compression_level: int = 9,
):
    """
    批量处理目录中的所有图片
    
    Args:
        input_dir (str): 输入图片目录
        blender_path (str): Blender可执行文件路径
        output_dir (str): 输出目录
        device (str): 渲染设备 ('cpu', 'cuda' or 'optix')
        resolution_x (int): 输出图像X分辨率
        resolution_y (int): 输出图像Y分辨率
        compression_level (int): PNG压缩级别(0-9)
    """
    input_dir = pth(input_dir)
    output_dir = pth(output_dir)
    
    if not input_dir.is_dir():
        raise ValueError(f"输入目录 {input_dir} 不存在")
    
    # 获取所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"警告: 在 {input_dir} 中没有找到图片文件")
        return
    
    for image_file in image_files:
        print(f"处理图片: {image_file}")
        # 为每个图片创建一个子目录
        img_output_dir = output_dir / image_file.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        process_image_to_3d(
            input_image_path=str(image_file),
            blender_path=blender_path,
            output_dir=img_output_dir,
            device=device,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            compression_level=compression_level
        )


if __name__ == "__main__":
    fire.Fire({
        'single': main,
        'batch': batch_process
    })