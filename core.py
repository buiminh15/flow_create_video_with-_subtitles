"""Logic dùng chung cho CLI và giao diện: OCR phụ đề cứng -> dịch -> dựng video.

Mọi toạ độ vùng tính trên khung chuẩn 1280x720 và tự scale theo video.
Thư mục dự án chứa: config.json, ocr_checkpoint.json, ocr.json, translated.<lang>.json, sub.<lang>.ass, clips/
"""
import json, subprocess, tempfile, time
from pathlib import Path

import cv2

REF_W, REF_H = 1280, 720
DEFAULT_CFG = {
    "sub_y0": 560, "sub_y1": 700,           # vùng phụ đề cũ
    "use_logo": True, "logo": [1050, 20, 220, 60],  # x, y, w, h
}
Progress = lambda frac, msg="": None  # callback mặc định


# ---------- tiện ích ----------
def probe(path):
    cap = cv2.VideoCapture(str(path))
    w, h, fps, n = int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7))
    cap.release()
    return w, h, fps, n


def project_dir(video, out_root="output"):
    d = Path(out_root).resolve() / Path(video).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cfg(pdir):
    p = Path(pdir) / "config.json"
    cfg = dict(DEFAULT_CFG)
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


def save_cfg(pdir, cfg):
    (Path(pdir) / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def ensure_ids(segs):
    for i, s in enumerate(segs):
        s["id"] = i + 1
    return segs


def tr_path(pdir, lang):
    return Path(pdir) / f"translated.{lang}.json"


# ---------- xem trước vùng ----------
def frame_with_boxes(video, t, cfg):
    """Khung hình tại giây t, vẽ ô vùng phụ đề (xanh) và logo (đỏ). Trả về ảnh RGB."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    sx, sy = w / REF_W, h / REF_H
    cv2.rectangle(frame, (int(w * 0.1), int(cfg["sub_y0"] * sy)), (int(w * 0.9), int(cfg["sub_y1"] * sy)), (0, 200, 0), 2)
    if cfg["use_logo"]:
        x, y, bw, bh = cfg["logo"]
        cv2.rectangle(frame, (int(x * sx), int(y * sy)), (int((x + bw) * sx), int((y + bh) * sy)), (0, 0, 230), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------- OCR ----------
def _similar(a, b):
    if a in b or b in a:
        return True
    common = sum(1 for ch in set(a) if ch in b)
    return common / min(len(set(a)), len(set(b))) > 0.7


def _merge(samples, step, dur):
    segs, cur = [], None
    for t, text in samples:
        if text and cur and _similar(cur["text"], text):
            cur["end"] = t + step
            if len(text) > len(cur["text"]):
                cur["text"] = text
        elif text:
            if cur:
                segs.append(cur)
            cur = {"start": t, "end": t + step, "text": text}
        else:
            if cur:
                segs.append(cur)
                cur = None
    if cur:
        segs.append(cur)
    segs = [s for s in segs if s["end"] - s["start"] >= step * 1.5 and len(s["text"]) > 1]
    for s in segs:
        s["start"], s["end"] = round(s["start"], 2), round(min(s["end"], dur), 2)
    return ensure_ids(segs)


def run_ocr(video, pdir, cfg, step=0.5, force=False, cb=Progress):
    """OCR vùng phụ đề, có checkpoint để chạy tiếp khi bị ngắt. Trả về list segment (đã có id)."""
    pdir = Path(pdir)
    final = pdir / "ocr.json"
    if final.exists() and not force:
        return ensure_ids(read_json(final))
    from rapidocr_onnxruntime import RapidOCR

    w, h, fps, n = probe(video)
    dur = n / fps
    stride = max(1, round(fps * step))
    y0, y1 = int(cfg["sub_y0"] * h / REF_H), int(cfg["sub_y1"] * h / REF_H)
    key = {"y0": cfg["sub_y0"], "y1": cfg["sub_y1"], "step": step, "n": n}
    ck_path = pdir / "ocr_checkpoint.json"
    ck = read_json(ck_path)
    samples, start_i = [], 0
    if ck and ck.get("key") == key and not force:
        samples, start_i = [tuple(x) for x in ck["samples"]], ck["next"]

    ocr = RapidOCR()
    cap = cv2.VideoCapture(str(video))
    if start_i:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_i)
    cw = w * 0.8

    def is_sub(r):  # chữ phụ đề: to, nằm giữa khung (chữ nền giao diện thì nhỏ / lệch)
        bh = r[0][3][1] - r[0][0][1]
        cx = (r[0][0][0] + r[0][1][0]) / 2
        return float(r[2]) > 0.6 and bh >= 28 * h / REF_H and abs(cx - cw / 2) < cw * 0.2

    prev_mask, prev_text = None, ""
    i = start_i
    while i < n:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                crop = frame[y0:y1, int(w * 0.1):int(w * 0.9)]
                # chữ phụ đề màu trắng: mặt nạ điểm sáng để bỏ qua OCR khi trống / không đổi (OCR rất chậm)
                mask = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 200
                if mask.sum() < mask.size * 0.0008:
                    text = ""
                elif prev_mask is not None and (mask & prev_mask).sum() / max(1, (mask | prev_mask).sum()) > 0.85:
                    text = prev_text
                else:
                    res, _ = ocr(crop)
                    lines = sorted((r for r in (res or []) if is_sub(r)), key=lambda r: r[0][0][1])
                    text = " ".join(r[1] for r in lines).strip()
                prev_mask, prev_text = mask, text
                samples.append((round(i / fps, 3), text))
                if len(samples) % 100 == 0:
                    write_json(ck_path, {"key": key, "next": i + 1, "samples": samples})
                    cb(i / n, f"OCR {i / fps:.0f}s / {dur:.0f}s")
        i += 1
    cap.release()
    segs = _merge(samples, step, dur)
    write_json(final, segs)
    ck_path.unlink(missing_ok=True)
    cb(1.0, f"OCR xong: {len(segs)} đoạn")
    return segs


# ---------- dịch ----------
def _engines(lang):
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    return [GoogleTranslator(source="zh-CN", target=lang),
            MyMemoryTranslator(source="zh-CN", target={"vi": "vi-VN"}.get(lang, lang))]


def _translate_one(engines, text):
    for eng in engines:
        for wait in (0, 3):
            try:
                time.sleep(wait or 0.3)
                out = eng.translate(text)
                if out:
                    return out
            except Exception:
                pass
    return ""


def translate_segments(segs, lang, only_missing=True, cb=Progress):
    """Dịch theo khối (gộp nhiều dòng một lần gọi). Dòng lỗi để tr="" chứ không giữ tiếng Trung.
    Trả về số dòng chưa dịch được."""
    todo = [s for s in segs if not (only_missing and s.get("tr"))]
    engines = _engines(lang)
    chunks, cur, size = [], [], 0
    for s in todo:
        if cur and (size + len(s["text"]) > 3500 or len(cur) >= 80):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s["text"]) + 1
    if cur:
        chunks.append(cur)

    for ci, chunk in enumerate(chunks):
        cb(ci / max(1, len(chunks)), f"Dịch khối {ci + 1}/{len(chunks)}")
        joined = "\n".join(s["text"] for s in chunk)
        out = None
        for wait in (0, 5, 15):
            time.sleep(wait)
            try:
                parts = engines[0].translate(joined).split("\n")
                if len(parts) == len(chunk):
                    out = parts
                break
            except Exception:
                continue
        if out:
            for s, t in zip(chunk, out):
                s["tr"] = t.strip()
        else:  # khối lệch dòng hoặc bị chặn -> dịch từng dòng
            for s in chunk:
                s["tr"] = _translate_one(engines, s["text"])
        time.sleep(1)
    cb(1.0, "Dịch xong")
    return sum(1 for s in segs if not s.get("tr"))


def export_for_manual(segs):
    """Bản gọn (id -> text) để tự dịch bằng AI bên ngoài."""
    return {str(s["id"]): s["text"] for s in segs}


def import_translation(segs, data):
    """Ghép {id: bản dịch} vào segs. Trả về (số dòng ghép được, danh sách id thiếu/rỗng, id lạ)."""
    data = {str(k): v for k, v in data.items()}
    ids = {str(s["id"]) for s in segs}
    n = 0
    for s in segs:
        v = data.get(str(s["id"]))
        if isinstance(v, str) and v.strip():
            s["tr"], n = v.strip(), n + 1
    missing = [s["id"] for s in segs if not s.get("tr")]
    return n, missing, [k for k in data if k not in ids]


# ---------- dựng video ----------
def _ts(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(segs, path, w, h, font="Arial", size=0):
    size = size or int(h * 0.055)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,60,60,{int(h * 0.06)},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    body = "".join(
        f"Dialogue: 0,{_ts(s['start'])},{_ts(s['end'])},Default,,0,0,0,,"
        f"{s['tr'].replace('{', '(').replace('}', ')').replace(chr(10), chr(92) + 'N')}\n"
        for s in segs if s.get("tr"))
    Path(path).write_text(head + body, encoding="utf-8-sig")


def _merge_intervals(segs, gap=0.3):
    iv = []
    for s in sorted(segs, key=lambda s: s["start"]):
        if iv and s["start"] - iv[-1][1] <= gap:
            iv[-1][1] = max(iv[-1][1], s["end"])
        else:
            iv.append([s["start"], s["end"]])
    return iv


def _time_expr(iv, lo=0, hi=None):
    """Biểu thức 'đang trong khoảng nào đó' dạng cây nhị phân (độ sâu log n) cho ffmpeg enable=."""
    hi = len(iv) if hi is None else hi
    if hi - lo == 0:
        return "0"
    if hi - lo == 1:
        return f"between(t,{iv[lo][0]:.2f},{iv[lo][1]:.2f})"
    mid = (lo + hi) // 2
    return f"if(lt(t,{iv[mid][0]:.2f}),{_time_expr(iv, lo, mid)},{_time_expr(iv, mid, hi)})"


def build_filter(segs, w, h, ass, cfg):
    sx, sy = w / REF_W, h / REF_H
    f, cur = [], "0:v"
    if cfg["use_logo"]:
        x, y, bw, bh = cfg["logo"]
        x, y = max(1, int(x * sx)), max(1, int(y * sy))
        bw, bh = min(int(bw * sx), w - x - 1), min(int(bh * sy), h - y - 1)
        f.append(f"[{cur}]delogo=x={x}:y={y}:w={bw}:h={bh}[d0]")
        cur = "d0"
    by0 = int(cfg["sub_y0"] * sy) + 20
    bh_ = max(8, int(cfg["sub_y1"] * sy) - 15 - by0)
    bx, bw_ = int(w * 0.1), int(w * 0.8)
    en = _time_expr(_merge_intervals(segs))
    f.append(f"[{cur}]split[a][b]")
    f.append(f"[b]crop={bw_}:{bh_}:{bx}:{by0},gblur=5:2[bl]")
    f.append(f"[a][bl]overlay={bx}:{by0}:enable='{en}'[o]")
    assp = str(Path(ass).resolve()).replace("\\", "/").replace(":", "\\:")
    f.append(f"[o]subtitles='{assp}'[v]")
    return ";\n".join(f)


def cut_points(segs, dur, n):
    """Điểm cắt gần mỗi n giây nhất, đặt vào khoảng lặng giữa hai câu (không cắt giữa câu)."""
    segs = sorted(segs, key=lambda s: s["start"])
    gaps = [(a["end"] + b["start"]) / 2 for a, b in zip(segs, segs[1:]) if b["start"] - a["end"] > 0.04]
    cuts, t = [], float(n)
    while t < dur - n * 0.3:
        near = [g for g in gaps if abs(g - t) <= n * 0.4 and g > (cuts[-1] if cuts else 0) + n * 0.3]
        c = min(near, key=lambda g: abs(g - t)) if near else t
        cuts.append(round(c, 2))
        t = c + n
    return cuts


def _run_ffmpeg(cmd, total, cb):
    p = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        if line.startswith("out_time_us="):
            try:
                cb(min(0.99, int(line.split("=")[1]) / 1e6 / total), "Đang encode")
            except ValueError:
                pass
        elif not line.startswith(("progress=", "frame=", "fps=", "stream_", "bitrate=", "total_size=",
                                  "out_time", "dup_frames=", "drop_frames=", "speed=")):
            tail.append(line)
    if p.wait() != 0:
        raise RuntimeError("ffmpeg lỗi:\n" + "".join(tail[-15:]))


QUALITY = {"nhanh": ("veryfast", 23), "cân bằng": ("medium", 20), "đẹp": ("slow", 18)}


def render(video, pdir, segs, lang, cfg, font="Arial", size=0, split_mode="sentence", split_sec=20,
           quality="cân bằng", test_seconds=0, cb=Progress):
    """Dựng video có phụ đề mới. split_mode: none | fixed | sentence. Trả về danh sách file."""
    pdir = Path(pdir)
    w, h, fps, n = probe(video)
    dur = n / fps
    ass = pdir / f"sub.{lang}.ass"
    write_ass(segs, ass, w, h, font, size)
    script = pdir / "filter.txt"
    script.write_text(build_filter(segs, w, h, ass, cfg), encoding="utf-8")
    preset, crf = QUALITY[quality]
    total = test_seconds or dur
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if test_seconds:
        cmd += ["-t", str(test_seconds)]
    cmd += ["-filter_complex_script", str(script), "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-c:a", "aac", "-b:a", "128k"]
    if test_seconds:
        out = pdir / f"test_{test_seconds}s.mp4"
        files = [out]
        cmd += [str(out)]
    elif split_mode == "none" or not split_sec:
        out = pdir / f"{Path(video).stem}.{lang}.mp4"
        files = [out]
        cmd += [str(out)]
    else:
        clips = pdir / "clips"
        clips.mkdir(exist_ok=True)
        for old in clips.glob("part_*.mp4"):
            old.unlink()
        cuts = cut_points(segs, dur, split_sec) if split_mode == "sentence" else \
            [float(split_sec) * k for k in range(1, int(dur // split_sec) + 1) if split_sec * k < dur - 1]
        if cuts:
            lst = ",".join(f"{c:.2f}" for c in cuts)
            cmd += ["-force_key_frames", lst, "-f", "segment", "-segment_times", lst,
                    "-reset_timestamps", "1", str(clips / "part_%03d.mp4")]
        else:
            cmd += [str(clips / "part_000.mp4")]
        files = None
    _run_ffmpeg(cmd, total, cb)
    if files is None:
        files = sorted((pdir / "clips").glob("part_*.mp4"))
    cb(1.0, "Xong")
    return [str(f) for f in files]


def preview_frame(video, pdir, segs, cfg, t, font="Arial", size=0):
    """Một khung hình tại giây t sau khi xử lý (logo, mờ, phụ đề mới). Trả về đường dẫn PNG."""
    pdir = Path(pdir)
    w, h, _, _ = probe(video)
    ass = pdir / "preview.ass"
    write_ass(segs, ass, w, h, font, size)
    script = pdir / "preview_filter.txt"
    script.write_text(build_filter(segs, w, h, ass, cfg), encoding="utf-8")
    out = pdir / "preview.png"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-copyts", "-i", str(video),
                        "-filter_complex_script", str(script), "-map", "[v]", "-frames:v", "1", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(r.stderr[-500:])
    return str(out)
