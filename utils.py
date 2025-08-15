from __future__ import annotations
from pathlib import Path
from typing import List, Union
import logging
import subprocess
import shutil
from datetime import datetime

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

from PIL import Image

logger = logging.getLogger(__name__)

def resolve_path_with_output_base(path: str) -> Path:
    p = Path(path)
    logger.info(f"Resolving path: {path}")
    logger.info(f"Is absolute: {p.is_absolute()}")
    if p.is_absolute():
        logger.info(f"Returning absolute path: {p}")
        return p
    # 获取ComfyUI根目录，通过查找custom_nodes目录
    cwd = Path.cwd()
    logger.info(f"Current working directory: {cwd}")
    # 检查是否在custom_nodes目录下
    if "custom_nodes" in str(cwd):
        # 找到ComfyUI根目录（custom_nodes的父目录）
        comfy_root = cwd
        while comfy_root.name != "custom_nodes" and comfy_root.parent != comfy_root:
            comfy_root = comfy_root.parent
        if comfy_root.parent != comfy_root:
            comfy_root = comfy_root.parent
    else:
        comfy_root = cwd
    output_dir = comfy_root / "output"
    result_path = output_dir / p
    logger.info(f"ComfyUI root: {comfy_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Final resolved path: {result_path}")
    return result_path

def resolve_time_pattern(path: str) -> str:
    import re
    now = datetime.now()
    # 处理 %yyyyMMdd HHmmss% 的旧格式
    if "%yyyyMMdd HHmmss%" in path:
        legacy = now.strftime("%Y%m%d %H%M%S")
        path = path.replace("%yyyyMMdd HHmmss%", legacy)
    
    # 处理 %xxx% 格式的时间占位符
    def replace_time_format(match):
        format_str = match.group(1)
        return now.strftime(format_str)
    
    # 查找所有 %xxx% 格式的占位符并替换
    path = re.sub(r'%([^%]+)%', replace_time_format, path)
    return path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _tensor_to_numpy_img(t) -> "np.ndarray":
    if torch is None:
        raise RuntimeError("torch is required to convert tensor images")
    if not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")
    logger.info(f"Input tensor shape: {t.shape}, dtype: {t.dtype}")
    
    # ComfyUI VAE解码输出的tensor形状为[batch, height, width, channels]
    x = t.detach().cpu()
    # 参考其他插件的实现方式
    # 修复squeeze问题，只移除第一个维度（batch维度），如果batch=1
    arr = np.clip(255. * x.numpy(), 0, 255).astype(np.uint8)
    # 如果是批量图像，只取最后一张；如果是单张图像，保持不变
    if arr.shape[0] == 1:
        arr = arr[0]
    elif arr.shape[0] > 1:
        # 取最后一张图像而不是第一张
        arr = arr[-1]
    logger.info(f"Output array shape: {arr.shape}, dtype: {arr.dtype}")
    return arr

def to_pil(img: Union[Image.Image, "np.ndarray", "torch.Tensor"]) -> Image.Image:
    logger.info(f"Converting to PIL, input type: {type(img)}")
    if isinstance(img, Image.Image):
        logger.info("Input is already PIL Image")
        return img
    if torch is not None and torch.is_tensor(img):
        logger.info("Input is torch.Tensor, converting...")
        arr = _tensor_to_numpy_img(img)
        pil_img = Image.fromarray(arr)
        logger.info(f"PIL Image created, size: {pil_img.size}, mode: {pil_img.mode}")
        return pil_img
    if np is not None and isinstance(img, np.ndarray):
        logger.info("Input is numpy array, converting...")
        a = np.clip(255. * img.squeeze(), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(a)
        logger.info(f"PIL Image created from numpy, size: {pil_img.size}, mode: {pil_img.mode}")
        return pil_img
    raise TypeError("Unsupported image type for to_pil")

def pil_to_preview_tensor(img: Image.Image):
    if np is None:
        raise RuntimeError("numpy is required")
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.astype("float32") / 255.0
    return arr

def pil_list_from_image_input(image) -> List[Image.Image]:
    if torch is not None and torch.is_tensor(image):
        x = image.detach().cpu()
        if x.ndim == 4:
            imgs = []
            for i in range(x.shape[0]):
                imgs.append(to_pil(x[i]))
            return imgs
        else:
            return [to_pil(x)]
    else:
        return [to_pil(image)]

class FrameSaver:
    def __init__(self, output_dir: str, filename_pattern: str = "frame_{index:04d}.png", fmt: str = "png"):
        dir_resolved_time = resolve_time_pattern(output_dir)
        self.output_dir = resolve_path_with_output_base(dir_resolved_time)
        self.filename_pattern = filename_pattern
        self.fmt = fmt.lower()
        ensure_dir(self.output_dir)
        self.index = self._load_last_index()
        # 检查filename_pattern是否包含index占位符
        self.has_index_placeholder = '{index' in filename_pattern

    def _load_last_index(self) -> int:
        # 只有当文件名模式包含index占位符时才加载最后的索引
        if not self.has_index_placeholder:
            return 0
            
        files = sorted(self.output_dir.glob("*"))
        if not files:
            return 0
        nums = []
        for f in files:
            stem = f.stem
            digits = "".join([c for c in stem if c.isdigit()])
            if digits:
                try:
                    nums.append(int(digits))
                except Exception:
                    pass
        return (max(nums) + 1) if nums else 0

    def save_pil(self, image: Image.Image) -> str:
        logger.info(f"Saving image to directory: {self.output_dir}")
        logger.info(f"Filename pattern: {self.filename_pattern}")
        logger.info(f"Has index placeholder: {self.has_index_placeholder}")
        logger.info(f"Current index: {self.index}")
        ensure_dir(self.output_dir)
        # 如果文件名不包含index占位符，则使用固定文件名
        if self.has_index_placeholder:
            # 支持两种格式: {index:04d} 和 %04d
            if '{index' in self.filename_pattern:
                filename = self.filename_pattern.format(index=self.index)
            else:  # %d 格式
                filename = self.filename_pattern % self.index
        else:
            filename = self.filename_pattern
        path = self.output_dir / filename
        logger.info(f"Full save path: {path}")
        try:
            image.save(str(path), format=self.fmt.upper())
            logger.info(f"Image saved successfully with format: {self.fmt.upper()}")
        except Exception as e:
            logger.warning("Save with explicit format failed (%s), fallback to auto: %s", self.fmt, e)
            try:
                image.save(str(path))
                logger.info("Image saved successfully with auto format")
            except Exception as e2:
                logger.error("Failed to save image: %s", e2)
                raise
        self.index += 1
        logger.info("Saved frame: %s", path)
        return str(path)

    def save_from_any(self, image_any) -> str:
        return self.save_pil(to_pil(image_any))

    def save_batch_from_any(self, images_any) -> List[str]:
        imgs = pil_list_from_image_input(images_any)
        # 批量保存时强制检查必须有占位符
        if not self.has_index_placeholder:
            raise ValueError("filename_pattern must contain a sequence placeholder like {index} or %d for batch saving")
        return [self.save_pil(im) for im in imgs]


def combine_frames_ffmpeg(frames_dir: str, output_path: str, fps: int = 30, codec: str = "libx264", crf: int = 23, frame_pattern: str = "frame_%04d.png", extra_flags: list | None = None, timeout: int | None = None):
    p = resolve_path_with_output_base(resolve_time_pattern(frames_dir))
    if not p.exists():
        raise FileNotFoundError(f"Frames directory not found: {p}")
    input_pattern = str(p / frame_pattern)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    logger.info("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def clean_frames_folder(frames_dir: str, keep_last: bool = True):
    p = resolve_path_with_output_base(resolve_time_pattern(frames_dir))
    if not p.exists():
        return
    files = sorted([f for f in p.iterdir() if f.is_file()])
    if not files:
        try:
            p.rmdir()
        except Exception:
            pass
        return
    if keep_last and len(files) > 0:
        for f in files[:-1]:
            try:
                f.unlink()
            except Exception as e:
                logger.warning("Failed to delete %s: %s", f, e)
    else:
        shutil.rmtree(p, ignore_errors=True)
