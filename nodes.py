from comfy.model import Node
from comfy.model.types import ImageType, StringType
from .utils import parse_time_tokens, save_comfyui_image, tensor_to_preview
import subprocess
import datetime
import os

class SaveSingleFrameToDisk(Node):
    title = "保存单帧到磁盘"
    type = "Rynode/LowMem"

    def inputs(self):
        return {
            "image": ImageType,
            "output_dir": StringType,
            "filename_pattern": StringType,
            "fmt": StringType
        }

    def outputs(self):
        return {
            "saved_path": StringType,
            "preview": ImageType
        }

    def process(self, image, output_dir, filename_pattern="frame.png", fmt="png"):
        dir_path = parse_time_tokens(output_dir)
        os.makedirs(dir_path, exist_ok=True)

        # 单帧可以是固定文件名
        save_path = os.path.join(dir_path, filename_pattern)
        save_comfyui_image(image, save_path, fmt)
        preview = tensor_to_preview(image)

        return save_path, preview


class SaveFrameBatchToDisk(Node):
    title = "批量保存帧到磁盘"
    type = "Rynode/LowMem"

    def inputs(self):
        return {
            "images": ImageType,
            "output_dir": StringType,
            "filename_pattern": StringType,
            "fmt": StringType
        }

    def outputs(self):
        return {
            "output_dir": StringType
        }

    def process(self, images, output_dir, filename_pattern="frame_%04d.png", fmt="png"):
        dir_path = parse_time_tokens(output_dir)
        os.makedirs(dir_path, exist_ok=True)

        # 校验 filename_pattern 必须包含序列占位符
        if "{index}" not in filename_pattern and "%04d" not in filename_pattern:
            raise ValueError("批量保存必须使用序列占位符，例如 frame_%04d.png 或 frame_{index}.png")

        for idx, img in enumerate(images):
            if "{index}" in filename_pattern:
                name = filename_pattern.format(index=idx+1)
            else:
                # 支持 %04d 样式
                name = filename_pattern % (idx+1)
            save_path = os.path.join(dir_path, name)
            save_comfyui_image(img, save_path, fmt)

        return dir_path

class FFmpegVideoCombineLowMem(Node):
    title = "FFmpeg 视频合成（低内存）"
    type = "Rynode/LowMem"

    def inputs(self):
        return {
            "frames_dir": StringType,
            "output_video_path": StringType,
            "fps": int,
            "codec": StringType,
            "crf": int,
            "frame_pattern": StringType,
            "delete_frames": bool,
            "keep_last_frame": bool
        }

    def outputs(self):
        return {
            "video_path": StringType
        }

    def process(
        self,
        frames_dir,
        output_video_path,
        fps=30,
        codec="libx264",
        crf=23,
        frame_pattern="frame_%04d.png",
        delete_frames=True,
        keep_last_frame=True
    ):
        # 替换时间占位符
        frames_dir = parse_time_tokens(frames_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = output_video_path.replace("{timestamp}", timestamp)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, frame_pattern),
            "-c:v", codec,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        subprocess.run(cmd, check=True)

        # 删除临时帧
        if delete_frames:
            import glob
            files = sorted(glob.glob(os.path.join(frames_dir, "*")))
            if keep_last_frame and files:
                files_to_delete = files[:-1]
            else:
                files_to_delete = files
            for f in files_to_delete:
                os.remove(f)

        return output_video_path