"""Giao diện web (Gradio) cho pipeline dịch phụ đề cứng. Chạy: python app.py"""
import json
import os
import time as time_module
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import core
import downloader
import gemini_ai
import youtube_trends
import video_cutter

load_dotenv()

COLS = ["id", "start", "end", "text", "tr"]
LANGS = {"Tiếng Việt": "vi", "English": "en", "Bahasa Indonesia": "id", "ไทย": "th"}


def _pd(video, out_root):
    if not video:
        raise gr.Error("Chưa chọn video.")
    return core.project_dir(video, out_root or "output")


def _rows(segs):
    return [[s["id"], s["start"], s["end"], s["text"], s.get("tr", "")] for s in segs]


def _segs_from_rows(rows):
    if hasattr(rows, "values"):  # pandas
        rows = rows.values.tolist()
    segs = []
    for r in rows:
        if r[0] in (None, "") or str(r[0]) == "nan":
            continue
        segs.append({"id": int(r[0]), "start": float(r[1]), "end": float(r[2]),
                     "text": str(r[3]), "tr": "" if str(r[4]) == "nan" else str(r[4] or "")})
    return segs


def _load_segs(pdir, lang):
    """Bản dịch (nếu có) ghi đè lên OCR; trả về list segment."""
    segs = core.read_json(core.tr_path(pdir, lang)) or core.read_json(Path(pdir) / "ocr.json")
    return core.ensure_ids(segs) if segs else []


def _cfg(sub_y0, sub_y1, use_logo, lx, ly, lw, lh, jp_mode=False, jp_y0=20, jp_y1=100):
    return {"sub_y0": int(sub_y0), "sub_y1": int(sub_y1), "use_logo": bool(use_logo),
            "logo": [int(lx), int(ly), int(lw), int(lh)],
            "jp_mode": bool(jp_mode), "jp_y0": int(jp_y0), "jp_y1": int(jp_y1)}


# ---------- tab 1 ----------
def on_video(path, out_root):
    """Chọn video: nạp cấu hình dự án + thông tin."""
    if not path:
        return [gr.update()] * 12
    w, h, fps, n = core.probe(path)
    pd = core.project_dir(path, out_root or "output")
    c = core.load_cfg(pd)
    info = f"{w}x{h}, {fps:.1f} fps, {n / fps:.0f}s  ·  dự án: {pd}"
    return (info, gr.update(maximum=max(1, n / fps), value=min(5, n / fps)), c["sub_y0"], c["sub_y1"],
            c["use_logo"], *c["logo"], c.get("jp_mode", False), c.get("jp_y0", 20), c.get("jp_y1", 100))


def show_regions(video, t, *cfg_args):
    if not video:
        return None
    return core.frame_with_boxes(video, t, _cfg(*cfg_args))


def do_ocr(video, out_root, step, force, *cfg_args, progress=gr.Progress()):
    pd = _pd(video, out_root)
    cfg = _cfg(*cfg_args)
    core.save_cfg(pd, cfg)
    segs = core.run_ocr(video, pd, cfg, step=step, force=force, cb=lambda f, m: progress(f, desc=m))
    return _rows(segs), f"OCR xong: {len(segs)} đoạn ({pd / 'ocr.json'})"


# ---------- tab 2 ----------
def do_translate(video, out_root, lang_name, only_missing, progress=gr.Progress()):
    pd, lang = _pd(video, out_root), LANGS[lang_name]
    segs = _load_segs(pd, lang)
    if not segs:
        raise gr.Error("Chưa có dữ liệu OCR.")
    miss = core.translate_segments(segs, lang, only_missing=only_missing,
                                   cb=lambda f, m: progress(f, desc=m))
    core.write_json(core.tr_path(pd, lang), segs)
    msg = f"Dịch xong {len(segs) - miss}/{len(segs)} đoạn."
    if miss:
        msg += f" ⚠ {miss} đoạn chưa dịch được (Google/MyMemory bị chặn?). Bấm Dịch lại sau ít phút hoặc sửa tay ở tab 3."
    return _rows(segs), msg


def do_export(video, out_root):
    pd = _pd(video, out_root)
    segs = core.read_json(Path(pd) / "ocr.json")
    if not segs:
        raise gr.Error("Chưa có dữ liệu OCR.")
    p = Path(pd) / "for_translate.json"
    core.write_json(p, core.export_for_manual(core.ensure_ids(segs)))
    return str(p)


def do_import(video, out_root, lang_name, file):
    pd, lang = _pd(video, out_root), LANGS[lang_name]
    if not file:
        raise gr.Error("Chưa chọn file bản dịch.")
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    segs = _load_segs(pd, lang)
    n, missing, unknown = core.import_translation(segs, data)
    core.write_json(core.tr_path(pd, lang), segs)
    msg = f"Ghép {n}/{len(segs)} đoạn."
    if missing:
        msg += f" ⚠ Thiếu/rỗng id: {missing[:30]}{'...' if len(missing) > 30 else ''}."
    if unknown:
        msg += f" ⚠ id lạ bị bỏ qua: {unknown[:10]}."
    return _rows(segs), msg


# ---------- tab 3 ----------
def load_table(video, out_root, lang_name):
    if not video:
        return []
    return _rows(_load_segs(_pd(video, out_root), LANGS[lang_name]))


def save_table(video, out_root, lang_name, rows):
    pd = _pd(video, out_root)
    segs = _segs_from_rows(rows)
    core.write_json(core.tr_path(pd, LANGS[lang_name]), segs)
    empty = sum(1 for s in segs if not s["tr"])
    return f"Đã lưu {len(segs)} đoạn" + (f" (⚠ {empty} đoạn chưa có bản dịch)" if empty else "")


def preview_row(video, out_root, lang_name, rows, font, size, *cfg_args, evt: gr.SelectData):
    if not video:
        return None
    r = evt.index[0]
    segs = _segs_from_rows(rows)
    t = segs[r]["start"] + 0.2
    return core.preview_frame(video, _pd(video, out_root), segs, _cfg(*cfg_args), t, font, int(size or 0))


# ---------- tab 4 ----------
def _render(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality, test, cfg_args,
            use_tts, voice, style, api_key, progress):
    pd = _pd(video, out_root)
    segs = _segs_from_rows(rows)
    if not segs or any(not s["tr"] for s in segs):
        raise gr.Error("Còn đoạn chưa có bản dịch — hoàn tất ở tab 2/3 trước.")
    dub_path = None
    if use_tts:
        progress(0, desc="Đang tạo giọng đọc TTS...")
        _, _, fps, n = core.probe(video)
        dub_path = pd / f"dub_{LANGS[lang_name]}_{voice}.wav"
        gemini_ai.dub_track(segs, api_key, voice, style, n / fps, dub_path,
                            cb=lambda f, m: progress(f * 0.5, desc=m))
    mode = {"Không cắt": "none", "Theo ranh giới câu": "sentence", "Cố định theo giây": "fixed"}[split_mode]
    files = core.render(video, pd, segs, LANGS[lang_name], _cfg(*cfg_args), font, int(size or 0), mode,
                        float(split_sec), quality, test_seconds=15 if test else 0, dub_audio=dub_path,
                        cb=lambda f, m: progress(0.5 + f * 0.5 if use_tts else f, desc=m))
    return files, files[0] if len(files) == 1 else None, f"Xong: {len(files)} file trong {pd}"


def render_test(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality,
                y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1,
                use_tts, voice, style, api_key, progress=gr.Progress()):
    return _render(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality, True,
                   (y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1), use_tts, voice, style, api_key,
                   progress)


def render_full(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality,
                y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1,
                use_tts, voice, style, api_key, progress=gr.Progress()):
    return _render(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality, False,
                   (y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1), use_tts, voice, style, api_key,
                   progress)


def tts_estimate(use_tts, rows):
    if not use_tts:
        return ""
    segs = _segs_from_rows(rows)
    n = sum(1 for s in segs if (s.get("tr") or s.get("text") or "").strip())
    return (f"⚠ Sẽ gọi Gemini TTS **{n} lần** (1 request/câu). Free tier chỉ **10 request/ngày** cho model "
            f"TTS — kiểm tra quota trước khi bấm Xuất, tránh chạy giữa đường bị lỗi 429.")


# ---------- tab 0: tải video ----------
def dl_info(url, cookies):
    if not url:
        raise gr.Error("Chưa nhập link.")
    try:
        i = downloader.fetch_info(url, cookies)
    except Exception as e:
        raise gr.Error(f"Không đọc được video: {str(e)[:300]}")
    mins = i["duration"] // 60
    lines = [f"**{i['title']}** · {i['count']} phần · {mins} phút"]
    if i["count"] > 1:
        lines += [f"- P{k + 1}: {t} ({d // 60}:{d % 60:02d})" for k, (t, d) in enumerate(i["parts"][:30])]
    return "\n".join(lines)


def dl_run(url, quality, cookies, progress=gr.Progress()):
    if not url:
        raise gr.Error("Chưa nhập link.")
    try:
        files = downloader.download(url, "downloads", quality, cookies, cb=lambda f, m: progress(f, desc=m))
    except Exception as e:
        raise gr.Error(f"Tải lỗi: {str(e)[:300]}")
    if not files:
        raise gr.Error("Không tải được file nào.")
    return (gr.update(choices=files, value=files[0]), files[0],
            f"Đã tải {len(files)} file vào thư mục downloads. Đã chọn file đầu tiên làm video, sang tab ① để OCR.")


def dl_update():
    return downloader.update_ytdlp()


# ---------- tab Gemini: transcribe & TTS ----------
def gm_transcribe(api_key, video, src_file, lang_hint, progress=gr.Progress()):
    path = src_file or video
    if not path:
        raise gr.Error("Chưa chọn video/audio để transcribe.")
    hint = "" if lang_hint in (None, "Tự nhận diện") else lang_hint
    segs = gemini_ai.transcribe(path, api_key, hint, cb=lambda f, m: progress(f, desc=m))
    if not segs:
        raise gr.Error("Không nhận diện được giọng nói nào.")
    return _rows(segs), f"Transcribe xong: {len(segs)} đoạn."


def gm_translate(api_key, lang_name, only_missing, rows, progress=gr.Progress()):
    segs = _segs_from_rows(rows)
    if not segs:
        raise gr.Error("Chưa có kết quả transcribe.")
    gemini_ai.translate_segments(segs, LANGS[lang_name], only_missing=only_missing, api_key=api_key,
                                 cb=lambda f, m: progress(f, desc=m))
    return _rows(segs), f"Dịch xong {len(segs)} đoạn sang {lang_name}."


def gm_use_as_ocr(video, out_root, src_file, lang_name, rows):
    path = video or src_file
    if not path:
        raise gr.Error("Chưa chọn video.")
    pd = core.project_dir(path, out_root or "output")
    segs = _segs_from_rows(rows)
    if not segs:
        raise gr.Error("Chưa có kết quả transcribe.")
    core.write_json(Path(pd) / "ocr.json", segs)
    core.write_json(core.tr_path(pd, LANGS[lang_name]), segs)
    msg = f"Đã lưu {len(segs)} đoạn vào {pd} — sang tab ③ Kiểm tra & sửa để xem/chỉnh, tab ④ để xuất video."
    is_video_file = Path(path).suffix.lower() in gemini_ai.VIDEO_EXTS
    return msg, (gr.update() if video or not is_video_file else path)


def gm_tts(api_key, video, out_root, text, voice, style, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Chưa nhập nội dung cần đọc.")
    out_dir = Path(out_root or "output")
    try:
        pd = _pd(video, out_root)
        out_dir = Path(pd)
    except gr.Error:
        out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"tts_{int(time_module.time())}.wav"
    path = gemini_ai.text_to_speech(text, api_key, voice, out, style, cb=lambda f, m: progress(f, desc=m))
    return path, f"Đã tạo file: {path}"


GM_PREVIEW_TEXT = "Xin chào, đây là mẫu thử giọng đọc. Một hai ba, thử giọng để chọn trước khi tạo file đầy đủ."


def gm_preview(api_key, voice, style, progress=gr.Progress()):
    import hashlib
    key = hashlib.md5(f"{voice}|{style}".encode("utf-8")).hexdigest()[:12]
    out = Path("output") / "_tts_preview_cache" / f"{voice}_{key}.wav"
    if out.exists():
        return str(out)
    path = gemini_ai.text_to_speech(GM_PREVIEW_TEXT, api_key, voice, out, style,
                                    cb=lambda f, m: progress(f, desc=m))
    return path


def gm_tts_from_table(api_key, video, out_root, rows, lang_name, voice, style, progress=gr.Progress()):
    segs = _segs_from_rows(rows)
    if not segs:
        raise gr.Error("Chưa có dữ liệu ở bảng ③.")
    text = "\n".join(s["tr"] or s["text"] for s in segs if (s["tr"] or s["text"]).strip())
    return gm_tts(api_key, video, out_root, text, voice, style, progress)


YT_COLS = ["title", "channel", "views", "likes", "comments", "published"]


def _yt_rows(items):
    return [[it["title"], it["channel"], it["views"], it["likes"], it["comments"], it["published"]]
            for it in items], [it["url"] for it in items]


def yt_categories(yt_key, region):
    try:
        cats = youtube_trends.video_categories(yt_key, region or "TH")
    except Exception as e:
        raise gr.Error(str(e)[:300])
    choices = ["(Tất cả)"] + [f"{v} ({k})" for k, v in sorted(cats.items(), key=lambda x: x[1])]
    return gr.update(choices=choices, value="(Tất cả)")


def yt_fetch(yt_key, region, mode, category, query, order, max_results):
    region = (region or "TH").strip().upper()
    try:
        if mode == "Top trending":
            cat_id = "" if not category or category.startswith("(") else category.split("(")[-1].rstrip(")")
            items = youtube_trends.trending(yt_key, region, cat_id, int(max_results))
        else:
            if not query.strip():
                raise gr.Error("Chưa nhập từ khoá tìm kiếm.")
            items = youtube_trends.search(yt_key, query.strip(), region, int(max_results), order)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e)[:300])
    rows, urls = _yt_rows(items)
    return rows, urls, f"Tìm được {len(items)} video (khu vực {region})."


def yt_pick(urls, evt: gr.SelectData):
    r = evt.index[0]
    if not urls or r >= len(urls):
        return gr.update()
    return urls[r]


CUTS_ROOT = Path("cuts")


def _cuts_dir(video):
    return CUTS_ROOT / Path(video).stem


def cut_load(video):
    if not video:
        return gr.update(value=None), gr.update(choices=[])
    files = video_cutter.list_cuts(_cuts_dir(video))
    return video, gr.update(choices=files, value=files[0] if files else None)


def cut_run(video, start, end, precise, progress=gr.Progress()):
    if not video:
        raise gr.Error("Chưa chọn video.")
    try:
        path = video_cutter.cut(video, start, end, _cuts_dir(video), bool(precise),
                                cb=lambda f, m: progress(f, desc=m))
    except ValueError as e:
        raise gr.Error(str(e))
    except RuntimeError as e:
        raise gr.Error(str(e)[:500])
    files = video_cutter.list_cuts(_cuts_dir(video))
    return gr.update(choices=files, value=path), path, f"Đã lưu: {path}"


def build():
    with gr.Blocks(title="Dịch phụ đề cứng") as ui:
        gr.Markdown("## Dịch phụ đề cứng (hardsub) · OCR → Dịch → Dựng video")
        with gr.Row():
            upload = gr.File(label="Chọn video", file_types=["video"], type="filepath", scale=2)
            video = gr.Textbox(label="Đường dẫn video (hoặc dán đường dẫn vào đây)", scale=3)
            out_root = gr.Textbox(label="Thư mục kết quả", value="output", scale=1)
        info = gr.Markdown()
        upload.change(lambda f: f or "", upload, video)

        with gr.Tabs():
            with gr.Tab("✂ Cắt video"):
                gr.Markdown(
                    "Cắt 1 đoạn từ video, lưu thành file riêng trong `cuts/<tên video>/`. "
                    "Phần này độc lập, không ảnh hưởng tới OCR/dịch/xuất video ở các tab khác."
                )
                with gr.Row():
                    cut_upload = gr.File(label="Chọn video", file_types=["video"], type="filepath", scale=1)
                    cut_src = gr.Textbox(label="Hoặc dán đường dẫn video", scale=2)
                cut_player = gr.Video(label="Xem video để xác định điểm cắt (tạm dừng rồi đọc thời gian)")
                with gr.Row():
                    cut_start = gr.Textbox(label="Bắt đầu (giây hoặc mm:ss, ví dụ 1:23)", value="0")
                    cut_end = gr.Textbox(label="Kết thúc (giây hoặc mm:ss)", value="10")
                    cut_precise = gr.Checkbox(value=True, label="Cắt chính xác (encode lại, chậm hơn)")
                cut_btn = gr.Button("Cắt & Lưu", variant="primary")
                cut_msg = gr.Markdown()
                cut_result = gr.Video(label="Đoạn vừa cắt", interactive=False)
                cut_list = gr.Dropdown(label="Các đoạn đã cắt từ video này", choices=[], interactive=True)

            with gr.Tab("Trend YouTube"):
                gr.Markdown(
                    "Xem video đang trending theo khu vực (YouTube Data API v3) để tìm nội dung phù hợp, "
                    "sau đó bấm vào một dòng để lấy link, dán sang tab ⓪ Tải video. "
                    "Cần `YOUTUBE_API_KEY` trong file .env hoặc nhập ở ô dưới."
                )
                yt_key = gr.Textbox(label="YouTube API key (bỏ trống nếu đã đặt trong .env)", type="password",
                                    value=os.environ.get("YOUTUBE_API_KEY", ""))
                with gr.Row():
                    yt_region = gr.Textbox(value="TH", label="Mã khu vực (TH=Thái Lan, VN, US, JP...)", scale=1)
                    yt_mode = gr.Radio(["Top trending", "Tìm theo từ khoá"], value="Top trending",
                                       label="Chế độ", scale=1)
                    yt_max = gr.Slider(5, 50, value=25, step=5, label="Số video", scale=1)
                with gr.Row():
                    yt_cat_btn = gr.Button("Tải danh mục")
                    yt_cat = gr.Dropdown(["(Tất cả)"], value="(Tất cả)", label="Danh mục (chỉ áp dụng Top trending)")
                with gr.Row():
                    yt_query = gr.Textbox(label="Từ khoá (chỉ áp dụng Tìm theo từ khoá)")
                    yt_order = gr.Dropdown(["viewCount", "date", "rating", "relevance"], value="viewCount",
                                           label="Sắp xếp theo")
                yt_fetch_btn = gr.Button("Lấy video", variant="primary")
                yt_msg = gr.Markdown()
                yt_tbl = gr.Dataframe(headers=YT_COLS, interactive=False, wrap=True,
                                      label="Bấm vào một dòng để lấy link video")
                yt_urls = gr.State([])
                yt_picked = gr.Textbox(label="Link video đã chọn (copy sang tab ⓪ Tải video)")

            with gr.Tab("⓪ Tải video"):
                gr.Markdown("Dán link Bilibili hoặc YouTube (tải tất cả các phần nếu video nhiều tập). Chỉ dùng cho nội dung bạn được phép tải.")
                dl_url = gr.Textbox(label="Link video Bilibili / YouTube")
                with gr.Row():
                    dl_quality = gr.Dropdown(list(downloader.QUALITY), value="1080p (H.264)", label="Chất lượng")
                    dl_cookies = gr.File(label="Cookies (tuỳ chọn, định dạng Netscape .txt)", type="filepath")
                with gr.Row():
                    dl_info_btn = gr.Button("Lấy thông tin")
                    dl_btn = gr.Button("Tải tất cả", variant="primary")
                    dl_upd_btn = gr.Button("Cập nhật yt-dlp")
                dl_msg = gr.Markdown()
                dl_pick = gr.Dropdown(label="File đã tải (chọn để dùng làm video)", choices=[], interactive=True)

            with gr.Tab("① OCR"):
                with gr.Row():
                    with gr.Column(scale=2):
                        t_sl = gr.Slider(0, 100, value=5, label="Xem khung hình tại giây")
                        img = gr.Image(label="Ô xanh = vùng phụ đề, ô đỏ = logo", interactive=False)
                    with gr.Column(scale=1):
                        y0 = gr.Slider(0, 719, value=560, step=1, label="Phụ đề: y trên (chuẩn 720p)")
                        y1 = gr.Slider(1, 720, value=700, step=1, label="Phụ đề: y dưới")
                        use_logo = gr.Checkbox(value=False, label="Xoá logo/watermark cố định")
                        lx = gr.Slider(0, 1279, value=1050, step=1, label="Logo x")
                        ly = gr.Slider(0, 719, value=20, step=1, label="Logo y")
                        lw = gr.Slider(1, 1280, value=220, step=1, label="Logo rộng")
                        lh = gr.Slider(1, 720, value=60, step=1, label="Logo cao")
                        jp_mode = gr.Checkbox(
                            value=False,
                            label="Video Nhật — double sub (OCR vùng trên = tiếng Nhật, xuất video 2 dòng: "
                                  "trên = text gốc trong file dịch, dưới = bản dịch)")
                        jp_y0 = gr.Slider(0, 719, value=20, step=1, label="Phụ đề Nhật: y trên",
                                         visible=False)
                        jp_y1 = gr.Slider(1, 720, value=100, step=1, label="Phụ đề Nhật: y dưới",
                                         visible=False)
                        step = gr.Slider(0.25, 1.0, value=0.5, step=0.25, label="Chu kỳ lấy mẫu OCR (giây)")
                        force = gr.Checkbox(label="OCR lại từ đầu (bỏ cache/checkpoint)")
                        ocr_btn = gr.Button("Chạy OCR", variant="primary")
                gr.Markdown(
                    "Ô xanh = vùng dịch (dưới) · Ô đỏ = logo · Ô vàng = vùng tiếng Nhật (trên, chỉ hiện khi bật "
                    "'Video Nhật'). Khi bật: OCR đọc vùng vàng làm 'text' (tiếng Nhật); lúc xuất video cả 2 vùng "
                    "đều được xoá nền gốc rồi chèn lại — dòng trên hiện 'text' (tiếng Nhật, từ file dịch json), "
                    "dòng dưới hiện 'tr' (bản dịch).")
                ocr_msg = gr.Markdown()
                ocr_tbl = gr.Dataframe(headers=COLS, label="Kết quả OCR", interactive=False, wrap=True)
            cfg_in = [y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1]

            with gr.Tab("② Dịch"):
                lang = gr.Dropdown(list(LANGS), value="Tiếng Việt", label="Ngôn ngữ đích")
                with gr.Row():
                    tr_btn = gr.Button("Dịch bằng Google (miễn phí, gộp theo khối)", variant="primary")
                    only_missing = gr.Checkbox(value=True, label="Chỉ dịch các dòng còn trống")
                tr_msg = gr.Markdown()
                with gr.Accordion("Dịch thủ công bằng AI bên ngoài (tuỳ chọn)", open=False):
                    exp_btn = gr.Button("Xuất file gọn (id → text tiếng Trung)")
                    exp_file = gr.File(label="File cần dịch", interactive=False)
                    imp_file = gr.File(label="Nhập file bản dịch (JSON {id: bản dịch})", type="filepath")
                    imp_btn = gr.Button("Nhập bản dịch")

            with gr.Tab("③ Kiểm tra & sửa"):
                gr.Markdown("Sửa trực tiếp cột *text*/*tr* hoặc thời gian. **Bấm vào một dòng** để xem khung hình có phụ đề mới.")
                with gr.Row():
                    reload_btn = gr.Button("Tải lại bảng")
                    save_btn = gr.Button("Lưu", variant="primary")
                edit_msg = gr.Markdown()
                with gr.Row():
                    tbl = gr.Dataframe(headers=COLS, datatype=["number", "number", "number", "str", "str"],
                                       interactive=True, wrap=True, scale=3, col_count=(5, "fixed"))
                    prev = gr.Image(label="Xem trước", interactive=False, scale=2)

            with gr.Tab("⑤ Gemini: Transcribe & TTS"):
                gr.Markdown(
                    "Dùng Gemini API (Google AI Studio) để nhận diện giọng nói thành văn bản và đọc văn bản "
                    "thành giọng nói. Lấy API key tại [aistudio.google.com/apikey](https://aistudio.google.com/apikey)."
                )
                gm_key = gr.Textbox(
                    label="API key Gemini (bỏ trống nếu đã đặt trong file .env)", type="password",
                    placeholder="AIza...",
                    value=os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", ""))
                with gr.Accordion("Transcribe (audio/video → văn bản)", open=True):
                    gm_src = gr.File(label="File audio/video (bỏ trống để dùng video đã chọn ở trên)",
                                     type="filepath")
                    gm_lang = gr.Dropdown(["Tự nhận diện", "Tiếng Việt", "English", "Chinese"],
                                          value="Tự nhận diện",
                                          label="Ngôn ngữ nói trong audio (giúp Gemini nhận diện đúng hơn; "
                                                "transcribe ra đúng ngôn ngữ này, không dịch)")
                    gm_tr_btn = gr.Button("Transcribe", variant="primary")
                    gm_tr_msg = gr.Markdown()
                    gm_tr_tbl = gr.Dataframe(headers=COLS, label="Kết quả transcribe", interactive=True, wrap=True)
                    with gr.Row():
                        gm_out_lang = gr.Dropdown(list(LANGS), value="Tiếng Việt", label="Dịch cột 'tr' sang")
                        gm_only_missing = gr.Checkbox(value=True, label="Chỉ dịch dòng còn trống")
                    gm_trans_btn = gr.Button("Dịch bằng Gemini (điền cột 'tr')", variant="primary")
                    gm_use_btn = gr.Button("Dùng kết quả này làm phụ đề + bản dịch (ghi ocr.json/translated.json)")
                with gr.Accordion("Text-to-Speech (văn bản → giọng nói)", open=True):
                    with gr.Row():
                        gm_voice = gr.Dropdown(gemini_ai.VOICES, value="Kore", label="Giọng đọc", scale=2)
                        gm_preview_btn = gr.Button("🔊 Nghe thử giọng", scale=1)
                    gm_preview_audio = gr.Audio(label="Mẫu thử", interactive=False)
                    gm_style = gr.Dropdown(gemini_ai.STYLE_PRESETS, value=gemini_ai.STYLE_PRESETS[0],
                                           allow_custom_value=True,
                                           label="Phong cách đọc (chọn mẫu hoặc tự nhập)")
                    gm_text = gr.Textbox(label="Văn bản cần đọc", lines=6)
                    with gr.Row():
                        gm_tts_btn = gr.Button("Đọc văn bản trên", variant="primary")
                        gm_tts_tbl_btn = gr.Button("Đọc theo bảng ③ (dùng bản dịch)")
                    gm_tts_audio = gr.Audio(label="Kết quả", interactive=False)
                    gm_tts_msg = gr.Markdown()

            with gr.Tab("④ Xuất video"):
                with gr.Row():
                    font = gr.Textbox(value="Arial", label="Font")
                    size = gr.Number(value=0, label="Cỡ chữ (0 = tự động)")
                    quality = gr.Radio(["nhanh", "cân bằng", "đẹp"], value="cân bằng", label="Chất lượng")
                with gr.Row():
                    split_mode = gr.Radio(["Theo ranh giới câu", "Cố định theo giây", "Không cắt"],
                                          value="Theo ranh giới câu", label="Cắt clip")
                    split_sec = gr.Number(value=20, label="Độ dài mỗi clip (giây)")
                use_tts = gr.Checkbox(
                    value=False,
                    label="Ghép giọng đọc TTS (Gemini) thay cho audio gốc — dùng giọng/phong cách đã chọn ở "
                          "tab ⑤ Gemini, lồng tiếng theo đúng thời điểm từng câu phụ đề")
                tts_est_msg = gr.Markdown()
                with gr.Row():
                    test_btn = gr.Button("Xuất thử 15 giây đầu")
                    full_btn = gr.Button("Xuất toàn bộ", variant="primary")
                out_msg = gr.Markdown()
                out_files = gr.File(label="File kết quả", file_count="multiple", interactive=False)
                out_vid = gr.Video(label="Xem thử", interactive=False)

        # ---- nối sự kiện ----
        cut_upload.change(lambda f: f or "", cut_upload, cut_src)
        cut_src.change(cut_load, cut_src, [cut_player, cut_list])
        cut_btn.click(cut_run, [cut_src, cut_start, cut_end, cut_precise], [cut_list, cut_result, cut_msg])
        cut_list.change(lambda p: p, cut_list, cut_result)

        yt_cat_btn.click(yt_categories, [yt_key, yt_region], yt_cat)
        yt_fetch_btn.click(yt_fetch, [yt_key, yt_region, yt_mode, yt_cat, yt_query, yt_order, yt_max],
                           [yt_tbl, yt_urls, yt_msg])
        yt_tbl.select(yt_pick, yt_urls, yt_picked)
        yt_picked.change(lambda u: u or "", yt_picked, dl_url)

        dl_info_btn.click(dl_info, [dl_url, dl_cookies], dl_msg)
        dl_btn.click(dl_run, [dl_url, dl_quality, dl_cookies], [dl_pick, video, dl_msg])
        dl_pick.input(lambda f: f or "", dl_pick, video)
        dl_upd_btn.click(dl_update, None, dl_msg)
        video.change(on_video, [video, out_root],
                    [info, t_sl, y0, y1, use_logo, lx, ly, lw, lh, jp_mode, jp_y0, jp_y1]).then(
            show_regions, [video, t_sl, *cfg_in], img).then(
            load_table, [video, out_root, lang], tbl)
        for c in [t_sl, *cfg_in]:
            c.release(show_regions, [video, t_sl, *cfg_in], img) if isinstance(c, gr.Slider) else \
                c.change(show_regions, [video, t_sl, *cfg_in], img)
        jp_mode.change(lambda on: (gr.update(visible=on), gr.update(visible=on)), jp_mode, [jp_y0, jp_y1])
        ocr_btn.click(do_ocr, [video, out_root, step, force, *cfg_in], [ocr_tbl, ocr_msg]).then(
            load_table, [video, out_root, lang], tbl)
        tr_btn.click(do_translate, [video, out_root, lang, only_missing], [tbl, tr_msg])
        exp_btn.click(do_export, [video, out_root], exp_file)
        imp_btn.click(do_import, [video, out_root, lang, imp_file], [tbl, tr_msg])
        reload_btn.click(load_table, [video, out_root, lang], tbl)
        save_btn.click(save_table, [video, out_root, lang, tbl], edit_msg)
        tbl.select(preview_row, [video, out_root, lang, tbl, font, size, *cfg_in], prev)
        render_in = [video, out_root, lang, tbl, font, size, split_mode, split_sec, quality]
        tts_in = [use_tts, gm_voice, gm_style, gm_key]
        test_btn.click(render_test, [*render_in, *cfg_in, *tts_in], [out_files, out_vid, out_msg])
        full_btn.click(render_full, [*render_in, *cfg_in, *tts_in], [out_files, out_vid, out_msg])
        use_tts.change(tts_estimate, [use_tts, tbl], tts_est_msg)
        tbl.change(tts_estimate, [use_tts, tbl], tts_est_msg)

        gm_tr_btn.click(gm_transcribe, [gm_key, video, gm_src, gm_lang], [gm_tr_tbl, gm_tr_msg])
        gm_trans_btn.click(gm_translate, [gm_key, gm_out_lang, gm_only_missing, gm_tr_tbl],
                           [gm_tr_tbl, gm_tr_msg])
        gm_use_btn.click(gm_use_as_ocr, [video, out_root, gm_src, gm_out_lang, gm_tr_tbl],
                         [gm_tr_msg, video]).then(load_table, [video, out_root, lang], tbl)
        gm_preview_btn.click(gm_preview, [gm_key, gm_voice, gm_style], gm_preview_audio)
        gm_tts_btn.click(gm_tts, [gm_key, video, out_root, gm_text, gm_voice, gm_style],
                        [gm_tts_audio, gm_tts_msg])
        gm_tts_tbl_btn.click(gm_tts_from_table, [gm_key, video, out_root, tbl, lang, gm_voice, gm_style],
                            [gm_tts_audio, gm_tts_msg])
    return ui


if __name__ == "__main__":
    build().queue().launch(inbrowser=True)
