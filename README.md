# ComfyUI-LowMemVideoSuite 插件

## 介绍

本插件提供低内存消耗的图像帧保存与视频合成功能，专为 ComfyUI 设计。  
通过将每帧图片逐个保存到本地，再利用系统命令行调用 FFmpeg 进行视频合成，避免传统 OpenCV 处理时的高内存占用。  

---

## 功能特点

- **单帧保存**：逐张保存采样器输出图片  
- **批量保存**：支持批量图片数组保存  
- **FFmpeg 视频合成**：按顺序合成视频，支持帧率、编码器、质量等配置  
- **支持路径时间格式**：目录和文件名支持日期时间占位符自动替换  
- **合成完成可删除临时帧**，支持保留最后一帧继续续接  
- **节点UI提供缩略图预览**

---

## 安装指南

### 1. 依赖

- Python 包：

```bash
pip install Pillow>=9.0.0 numpy torch
```

系统工具：

请自行安装并配置 FFmpeg，确保命令行可用。
Windows 推荐下载静态包，添加至系统 PATH；Linux/macOS 可通过包管理器安装。

### 2. 部署
将插件目录放入 ComfyUI custom_nodes 目录，如：

```
ComfyUI/custom_nodes/ComfyUI-LowMemVideoSuite/
```

## 使用说明

### 节点分类
位于 Rynode/LowMem 分类下。

### 节点详情

#### 1. 保存单帧到磁盘 / SaveSingleFrameToDisk
##### 输入

- `image`: 输入图片
- `output_dir`: 保存目录，支持时间格式，如 `./frames/%Y%m%d_%H%M%S` 或 `./output/%Y%m%d%`
- `filename_pattern`: 文件名模板，可以是固定文件名如 `cover.png`，也可以是带占位符的格式如 `frame_{index:04d}.png`
- `fmt`: 图片格式，默认 png

##### 输出

- `path`: 保存的图片路径
- `preview`: 预览缩略图

#### 2. 批量保存帧到磁盘 / SaveFrameBatchToDisk
##### 输入

- `images`: 图片列表
- `output_dir`: 保存目录，支持时间格式，如 `./frames/%Y%m%d_%H%M%S` 或 `./output/%Y%m%d%`
- `filename_pattern`: 文件名模板，必须带序列占位符（`%04d` 或 `{index}`），如 `frame_%04d.png` 或 `frame_{index:04d}.png`
- `fmt`: 图片格式，默认 png

##### 输出

- `paths`: 图片路径列表
- `preview`: 最后一帧缩略图

#### 3. FFmpeg 视频合成（低内存） / FFmpegVideoCombineLowMem
##### 输入

- `frames_dir`: 帧图片目录，支持时间格式
- `output_video_path`: 视频输出路径，支持 `{timestamp}` 替换为当前时间
- `fps`: 视频帧率，默认30
- `codec`: 视频编码，默认 libx264
- `crf`: 质量参数，默认23
- `frame_pattern`: 图片帧名匹配，默认 `frame_%04d.png`
- `delete_frames`: 是否合成后删除帧文件，默认 true
- `keep_last_frame`: 删除时是否保留最后一帧，默认 true

##### 输出

- `video_path`: 生成视频完整路径

## 使用流程示例

1. 采样生成或处理图片帧
2. 使用"批量保存帧到磁盘"节点，保存图片到指定目录（支持时间变量自动创建目录）
3. 调用"FFmpeg 视频合成（低内存）"节点，将保存帧目录传入，配置输出路径及参数
4. 视频合成完成，自动删除临时帧（可选择保留最后一帧）

## 常见问题

### ffmpeg 找不到命令
请确认系统已正确安装 ffmpeg，并且 ffmpeg 可在命令行中直接运行。

### 保存路径无效或文件没生成
确认 output_dir 是否正确，有权限写入，支持时间格式的路径会自动替换。

### 视频播放异常或无视频
确认帧图片命名连续且符合 frame_pattern，fps 设置合理。