import os
from pathlib import Path
from PIL import Image
from .utils import save_frame, combine_frames_ffmpeg, clean_frames

class SaveSingleFrameToDisk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": "./frames"}),
                "filename_pattern": ("STRING", {"default": "frame_{index:04d}.png"}),
                "format": ("STRING", {"default": "png"})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_single_frame"
    CATEGORY = "Video/LowMem"

    def save_single_frame(self, image, output_dir, filename_pattern, format):
        output_dir = Path(output_dir)
        existing_files = sorted(output_dir.glob("*"))
        index = len(existing_files) + 1
        path = save_frame(image, output_dir, index, filename_pattern, format)
        return (str(path),)


class FFmpegVideoCombineLowMem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "./frames"}),
                "output_path": ("STRING", {"default": "./output.mp4"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 240}),
                "frame_pattern": ("STRING", {"default": "frame_%04d.png"}),
                "codec": ("STRING", {"default": "libx264"}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51}),
                "keep_last_frame": ("BOOL", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine_video"
    CATEGORY = "Video/LowMem"

    def combine_video(self, frames_dir, output_path, fps, frame_pattern, codec, crf, keep_last_frame):
        combine_frames_ffmpeg(frames_dir, output_path, fps, codec, crf, frame_pattern)
        clean_frames(frames_dir, keep_last_frame)
        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "SaveSingleFrameToDisk": SaveSingleFrameToDisk,
    "FFmpegVideoCombineLowMem": FFmpegVideoCombineLowMem
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveSingleFrameToDisk": "Save Single Frame To Disk",
    "FFmpegVideoCombineLowMem": "FFmpeg Video Combine (Low Memory)"
}
