from __future__ import annotations
from typing import Optional, List, Any
import logging

from PIL import Image

from .utils import (
    FrameSaver,
    combine_frames_ffmpeg,
    clean_frames_folder,
    resolve_time_pattern,
    pil_list_from_image_input,
    pil_to_preview_tensor,
    to_pil,
)

logger = logging.getLogger(__name__)

class SaveSingleFrameToDisk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "./frames/%Y%m%d_%H%M%S"}),
                "filename_pattern": ("STRING", {"default": "frame.png"}),
                "fmt": ("STRING", {"default": "png"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("path", "preview")
    FUNCTION = "save_single"
    CATEGORY = "Rynode/LowMem"

    def save_single(self, image, output_dir: str = "./frames/%Y%m%d_%H%M%S", filename_pattern: str = "frame.png", fmt: str = "png"):
        # 对于单帧保存，可以直接使用FrameSaver
        saver = FrameSaver(output_dir, filename_pattern, fmt)
        preview_img = to_pil(image)
        path = saver.save_pil(preview_img)
        preview = pil_to_preview_tensor(preview_img)
        logger.info(f"[SaveSingleFrameToDisk] Saved: {path}")
        return (path, preview,)


class SaveFrameBatchToDisk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "./frames/%Y%m%d_%H%M%S"}),
                "filename_pattern": ("STRING", {"default": "frame_{index:04d}.png"}),
                "fmt": ("STRING", {"default": "png"}),
            },
        }

    RETURN_TYPES = ("ARRAY", "IMAGE")
    RETURN_NAMES = ("paths", "preview")
    FUNCTION = "save_batch"
    CATEGORY = "Rynode/LowMem"

    def save_batch(self, images, output_dir: str = "./frames/%Y%m%d_%H%M%S", filename_pattern: str = "frame_{index:04d}.png", fmt: str = "png"):
        # 检查filename_pattern是否包含序列占位符
        if "{index" not in filename_pattern and "%d" not in filename_pattern:
            raise ValueError("filename_pattern must contain a sequence placeholder like {index} or %d")
            
        saver = FrameSaver(output_dir, filename_pattern, fmt)
        pil_list = pil_list_from_image_input(images)
        paths = []
        for im in pil_list:
            paths.append(saver.save_pil(im))
        preview = pil_to_preview_tensor(pil_list[-1])
        logger.info(f"[SaveFrameBatchToDisk] Saved {len(paths)} frames to {resolve_time_pattern(output_dir)}")
        return (paths, preview,)


class FFmpegVideoCombineLowMem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "./frames/%Y%m%d_%H%M%S"}),
                "output_video_path": ("STRING", {"default": "./output/video_{timestamp}.mp4"}),
            },
            "optional": {
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "codec": ("STRING", {"default": "libx264"}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51}),
                "frame_pattern": ("STRING", {"default": "frame_{index:04d}.png"}),
                "delete_frames": ("BOOL", {"default": True}),
                "keep_last_frame": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "combine_video"
    CATEGORY = "Rynode/LowMem"

    def combine_video(
        self,
        frames_dir: str = "./frames/%Y%m%d_%H%M%S",
        output_video_path: str = "./output/video_{timestamp}.mp4",
        fps: int = 30,
        codec: str = "libx264",
        crf: int = 23,
        frame_pattern: str = "frame_%04d.png",
        delete_frames: bool = True,
        keep_last_frame: bool = True,
    ):
        # 解析时间格式，支持自定义 timestamp 标记
        from datetime import datetime
        now = datetime.now()
        if "{timestamp}" in output_video_path:
            output_video_path = output_video_path.replace("{timestamp}", now.strftime("%Y%m%d_%H%M%S"))
        frames_dir_resolved = resolve_time_pattern(frames_dir)

        # 绝对路径修正
        from pathlib import Path
        frames_dir_path = Path(frames_dir_resolved).expanduser().resolve()
        output_video_path = Path(output_video_path).expanduser().resolve()

        logger.info(f"[FFmpegVideoCombineLowMem] Combining frames from {frames_dir_path} to video {output_video_path}")

        combine_frames_ffmpeg(
            str(frames_dir_path),
            str(output_video_path),
            fps=fps,
            codec=codec,
            crf=crf,
            frame_pattern=frame_pattern,
        )

        if delete_frames:
            clean_frames_folder(str(frames_dir_path), keep_last=keep_last_frame)

        return (str(output_video_path),)


NODE_CLASS_MAPPINGS = {
    "保存单帧到磁盘 / SaveSingleFrameToDisk": SaveSingleFrameToDisk,
    "批量保存帧到磁盘 / SaveFrameBatchToDisk": SaveFrameBatchToDisk,
    "FFmpeg 视频合成（低内存） / FFmpegVideoCombineLowMem": FFmpegVideoCombineLowMem,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    SaveSingleFrameToDisk: "保存单帧到磁盘 / SaveSingleFrameToDisk",
    SaveFrameBatchToDisk: "批量保存帧到磁盘 / SaveFrameBatchToDisk",
    FFmpegVideoCombineLowMem: "FFmpeg 视频合成（低内存） / FFmpegVideoCombineLowMem",
}
