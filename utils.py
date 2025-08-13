import os
import re
import datetime
import torch
import numpy as np
from PIL import Image

def parse_time_tokens(path_template: str) -> str:
    """
    将路径中 %yyyyMMdd%、%HHmmss% 这种占位符替换为当前时间
    """
    now = datetime.datetime.now()
    replacements = {
        "yyyy": now.strftime("%Y"),
        "MM": now.strftime("%m"),
        "dd": now.strftime("%d"),
        "HH": now.strftime("%H"),
        "mm": now.strftime("%M"),
        "ss": now.strftime("%S"),
    }

    def replacer(match):
        key = match.group(1)
        return replacements.get(key, match.group(0))

    return re.sub(r"%([a-zA-Z]+)%", replacer, path_template)

def save_comfyui_image(tensor, output_path, fmt="png"):
    """
    保存 ComfyUI Tensor 图片到磁盘，并返回保存路径
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # 如果是批量，取第一张
    if tensor.ndim == 4:
        tensor = tensor[0]

    # 转换 0~1 float -> 0~255 uint8
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)

    # 转换为 PIL.Image
    if tensor.shape[2] == 1:
        img = Image.fromarray(tensor[:, :, 0], mode="L")
    else:
        img = Image.fromarray(tensor, mode="RGB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format=fmt.upper())
    return output_path

def tensor_to_preview(tensor, size=(256, 256)):
    """
    将 ComfyUI Tensor 转为缩略图 tensor，用于节点预览
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    if tensor.ndim == 4:
        tensor = tensor[0]

    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)

    if tensor.shape[2] == 1:
        img = Image.fromarray(tensor[:, :, 0], mode="L")
    else:
        img = Image.fromarray(tensor, mode="RGB")

    img.thumbnail(size)
    preview = np.array(img).astype(np.float32) / 255.0
    if preview.ndim == 2:
        preview = np.expand_dims(preview, axis=2)
    preview = np.expand_dims(preview, axis=0)  # batch=1
    preview_tensor = torch.from_numpy(preview)
    return preview_tensor
