import subprocess
from pathlib import Path

def save_frame(image, output_dir, index, filename_pattern, fmt):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_pattern.format(index=index)
    filepath = output_dir / filename
    if hasattr(image, 'save'):
        image.save(str(filepath), format=fmt.upper())
    else:
        raise ValueError("Input is not a valid image object with save method")
    return filepath

def combine_frames_ffmpeg(frames_dir, output_path, fps, codec, crf, frame_pattern):
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory {frames_dir} does not exist")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / frame_pattern),
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def clean_frames(frames_dir, keep_last):
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob("*"))
    if keep_last and frames:
        for f in frames[:-1]:
            f.unlink()
    else:
        for f in frames:
            f.unlink()
        frames_dir.rmdir()
