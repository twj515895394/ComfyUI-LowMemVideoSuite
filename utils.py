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
    now = datetime.now()
    if "%yyyyMMdd HHmmss%" in path:
        legacy = now.strftime("%Y%m%d %H%M%S")
        path = path.replace("%yyyyMMdd HHmmss%", legacy)
    return now.strftime(path)

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

    def _load_last_index(self) -> int:
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
        filename = self.filename_pattern.format(index=self.index)
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
