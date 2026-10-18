"""Tải video Bilibili bằng yt-dlp (tải tất cả các phần nếu video nhiều tập)."""
import re
import subprocess
import sys
from pathlib import Path

import yt_dlp

QUALITY = {
    "1080p (H.264)": "bv*[height<=1080][vcodec^=avc]+ba/bv*[height<=1080]+ba/b[height<=1080]",
    "720p (H.264)": "bv*[height<=720][vcodec^=avc]+ba/bv*[height<=720]+ba/b[height<=720]",
    "480p (H.264)": "bv*[height<=480][vcodec^=avc]+ba/bv*[height<=480]+ba/b[height<=480]",
    "Tốt nhất có thể": "bv*+ba/b",
}


def clean_url(url):
    """Bỏ tham số theo dõi, giữ lại ?p=N nếu có."""
    url = url.strip()
    m = re.search(r"(https?://(?:www\.)?bilibili\.com/video/[A-Za-z0-9]+)/?(\?[^#]*)?", url)
    if not m:
        return url
    p = re.search(r"[?&]p=(\d+)", m.group(2) or "")
    return m.group(1) + "/" + (f"?p={p.group(1)}" if p else "")


def _opts(cookies=None, **extra):
    o = {"quiet": True, "no_warnings": True, "noprogress": True, "windowsfilenames": True,
         "socket_timeout": 60, "retries": 15, "fragment_retries": 15, "continuedl": True,
         "http_chunk_size": 5242880, **extra}  # 5 MB chunks để tránh timeout trên mạng chậm
    if cookies:
        o["cookiefile"] = str(cookies)
    return o


def fetch_info(url, cookies=None):
    """Trả về dict: title, count, duration (giây, tổng), parts [(tên, giây)]."""
    with yt_dlp.YoutubeDL(_opts(cookies, extract_flat=False, skip_download=True)) as ydl:
        info = ydl.extract_info(clean_url(url), download=False)
    entries = list(info.get("entries") or [info])
    parts = [(e.get("title") or "?", int(e.get("duration") or 0)) for e in entries]
    return {"title": info.get("title") or "?", "count": len(entries),
            "duration": sum(d for _, d in parts), "parts": parts}


def download(url, out_dir, quality, cookies=None, cb=lambda f, m: None):
    """Tải tất cả các phần vào out_dir. Trả về danh sách đường dẫn file .mp4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"i": 1, "n": 1}
    files = []

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = d["downloaded_bytes"] / total if total else 0
            cb((state["i"] - 1 + frac) / state["n"], f"Tải phần {state['i']}/{state['n']}: {frac:.0%}")
        elif d["status"] == "finished":
            cb(state["i"] / state["n"], f"Phần {state['i']}/{state['n']}: đang ghép hình + tiếng")

    def post_hook(path):
        files.append(path)
        state["i"] += 1

    opts = _opts(cookies, format=QUALITY[quality], merge_output_format="mp4", progress_hooks=[hook],
                 outtmpl=str(out_dir / "%(title).60s [%(id)s]" / "%(playlist_index|1)02d_%(title).50s.%(ext)s"),
                 noplaylist=False)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.add_post_hook(post_hook)
        meta = ydl.extract_info(clean_url(url), download=False)
        state["n"] = len(list(meta.get("entries") or [meta]))
        ydl.download([clean_url(url)])
    cb(1.0, f"Tải xong {len(files)} file")
    return sorted(set(files))


def update_ytdlp():
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "xong"
