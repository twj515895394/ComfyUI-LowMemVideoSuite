# ComfyUI-LowMemVideoSuite

## 概述
这是一个为 ComfyUI 设计的低内存视频合成插件，使用 FFmpeg 将磁盘帧图合成视频，避免一次性加载所有帧到内存。

## 功能
- SaveSingleFrameToDisk: 保存单帧到磁盘。
- FFmpegVideoCombineLowMem: 使用 FFmpeg 合成视频并可选清理帧。

## 安装
1. 将本项目放到 `ComfyUI/custom_nodes/` 目录。
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 确保系统已安装 FFmpeg 并可在命令行调用。

## 使用
在 ComfyUI 中添加对应节点，根据需要保存帧并合成视频。
