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
    if p.is_absolute():
        return p
    cwd = Path.cwd()
    comfy_root = cwd
    output_dir = comfy_root / "output"
    return output_dir / p

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
    x = t.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor of 3 dims (HWC/CHW), got shape {tuple(x.shape)}")
    if x.shape[0] in (1,3) and x.shape[-1] not in (1,3):
        x = x.permute(1,2,0)
    a = x.numpy()
    a = a.clip(0,1)
    a = (a * 255.0).round().astype("uint8")
    if a.shape[-1] == 1:
        a = a[..., 0]
    return a

def to_pil(img: Union[Image.Image, "np.ndarray", "torch.Tensor"]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if torch is not None and torch.is_tensor(img):
        arr = _tensor_to_numpy_img(img)
        return Image.fromarray(arr)
    if np is not None and isinstance(img, np.ndarray):
        a = img
        if a.dtype != "uint8":
            a = a.clip(0,1)
            a = (a * 255.0).round().astype("uint8")
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[...,0]
        return Image.fromarray(a)
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
        try:
            image.save(str(path), format=self.fmt.upper())
        except Exception as e:
            logger.warning("Save with explicit format failed (%s), fallback to auto: %s", self.fmt, e)
            image.save(str(path))
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