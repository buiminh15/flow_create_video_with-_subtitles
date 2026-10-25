"""Cắt 1 đoạn từ video gốc, lưu file riêng. Độc lập với pipeline OCR/dịch/TTS."""
import re
import subprocess
from pathlib import Path


def parse_time(s):
    """'1:23' / '1:23.5' / '83' -> giây (float)."""
    s = str(s).strip()
    if not s:
        return 0.0
    m = re.match(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$", s)
    if not m:
        raise ValueError(f"Thời gian không hợp lệ: {s}")
    mins, secs = m.groups()
    return (int(mins) * 60 if mins else 0) + float(secs)


def fmt_time(sec):
    sec = max(0, sec)
    return f"{int(sec // 60)}m{sec % 60:04.1f}s".replace(".", "_")


def cut(video, start, end, out_dir, precise=True, cb=None):
    """Cắt [start, end) (giây hoặc 'mm:ss') từ video, lưu vào out_dir. Trả về đường dẫn file."""
    start_s, end_s = parse_time(start), parse_time(end)
    if end_s <= start_s:
        raise ValueError("Thời điểm kết thúc phải lớn hơn thời điểm bắt đầu.")
    dur = end_s - start_s
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{Path(video).stem}_{fmt_time(start_s)}-{fmt_time(end_s)}.mp4"
    out = out_dir / name
    if cb:
        cb(0.1, f"Đang cắt {start_s:.1f}s -> {end_s:.1f}s...")
    if precise:
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", str(out)]
    else:  # nhanh (copy stream), có thể lệch tới khung hình khoá gần nhất
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
               "-c", "copy", "-avoid_negative_ts", "make_zero", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("ffmpeg lỗi:\n" + r.stderr[-800:])
    if cb:
        cb(1.0, "Xong")
    return str(out)


def list_cuts(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    return sorted((str(p) for p in out_dir.glob("*.mp4")), key=lambda p: Path(p).stat().st_mtime, reverse=True)
