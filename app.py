"""Giao diện web (Gradio) cho pipeline dịch phụ đề cứng. Chạy: python app.py"""
import json
from pathlib import Path

import gradio as gr

import core
import downloader

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


def _cfg(sub_y0, sub_y1, use_logo, lx, ly, lw, lh):
    return {"sub_y0": int(sub_y0), "sub_y1": int(sub_y1), "use_logo": bool(use_logo),
            "logo": [int(lx), int(ly), int(lw), int(lh)]}


# ---------- tab 1 ----------
def on_video(path, out_root):
    """Chọn video: nạp cấu hình dự án + thông tin."""
    if not path:
        return [gr.update()] * 9
    w, h, fps, n = core.probe(path)
    pd = core.project_dir(path, out_root or "output")
    c = core.load_cfg(pd)
    info = f"{w}x{h}, {fps:.1f} fps, {n / fps:.0f}s  ·  dự án: {pd}"
    return (info, gr.update(maximum=max(1, n / fps), value=min(5, n / fps)), c["sub_y0"], c["sub_y1"],
            c["use_logo"], *c["logo"])


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
            progress):
    pd = _pd(video, out_root)
    segs = _segs_from_rows(rows)
    if not segs or any(not s["tr"] for s in segs):
        raise gr.Error("Còn đoạn chưa có bản dịch — hoàn tất ở tab 2/3 trước.")
    mode = {"Không cắt": "none", "Theo ranh giới câu": "sentence", "Cố định theo giây": "fixed"}[split_mode]
    files = core.render(video, pd, segs, LANGS[lang_name], _cfg(*cfg_args), font, int(size or 0), mode,
                        float(split_sec), quality, test_seconds=15 if test else 0,
                        cb=lambda f, m: progress(f, desc=m))
    return files, files[0] if len(files) == 1 else None, f"Xong: {len(files)} file trong {pd}"


def render_test(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality,
                y0, y1, use_logo, lx, ly, lw, lh, progress=gr.Progress()):
    return _render(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality, True,
                   (y0, y1, use_logo, lx, ly, lw, lh), progress)


def render_full(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality,
                y0, y1, use_logo, lx, ly, lw, lh, progress=gr.Progress()):
    return _render(video, out_root, lang_name, rows, font, size, split_mode, split_sec, quality, False,
                   (y0, y1, use_logo, lx, ly, lw, lh), progress)


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
            with gr.Tab("⓪ Tải video"):
                gr.Markdown("Dán link Bilibili (tải tất cả các phần nếu video nhiều tập). Chỉ dùng cho nội dung bạn được phép tải.")
                dl_url = gr.Textbox(label="Link video Bilibili")
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
                        use_logo = gr.Checkbox(value=True, label="Xoá logo/watermark cố định")
                        lx = gr.Slider(0, 1279, value=1050, step=1, label="Logo x")
                        ly = gr.Slider(0, 719, value=20, step=1, label="Logo y")
                        lw = gr.Slider(1, 1280, value=220, step=1, label="Logo rộng")
                        lh = gr.Slider(1, 720, value=60, step=1, label="Logo cao")
                        step = gr.Slider(0.25, 1.0, value=0.5, step=0.25, label="Chu kỳ lấy mẫu OCR (giây)")
                        force = gr.Checkbox(label="OCR lại từ đầu (bỏ cache/checkpoint)")
                        ocr_btn = gr.Button("Chạy OCR", variant="primary")
                ocr_msg = gr.Markdown()
                ocr_tbl = gr.Dataframe(headers=COLS, label="Kết quả OCR", interactive=False, wrap=True)
            cfg_in = [y0, y1, use_logo, lx, ly, lw, lh]

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

            with gr.Tab("④ Xuất video"):
                with gr.Row():
                    font = gr.Textbox(value="Arial", label="Font")
                    size = gr.Number(value=0, label="Cỡ chữ (0 = tự động)")
                    quality = gr.Radio(["nhanh", "cân bằng", "đẹp"], value="cân bằng", label="Chất lượng")
                with gr.Row():
                    split_mode = gr.Radio(["Theo ranh giới câu", "Cố định theo giây", "Không cắt"],
                                          value="Theo ranh giới câu", label="Cắt clip")
                    split_sec = gr.Number(value=20, label="Độ dài mỗi clip (giây)")
                with gr.Row():
                    test_btn = gr.Button("Xuất thử 15 giây đầu")
                    full_btn = gr.Button("Xuất toàn bộ", variant="primary")
                out_msg = gr.Markdown()
                out_files = gr.File(label="File kết quả", file_count="multiple", interactive=False)
                out_vid = gr.Video(label="Xem thử", interactive=False)

        # ---- nối sự kiện ----
        dl_info_btn.click(dl_info, [dl_url, dl_cookies], dl_msg)
        dl_btn.click(dl_run, [dl_url, dl_quality, dl_cookies], [dl_pick, video, dl_msg])
        dl_pick.input(lambda f: f or "", dl_pick, video)
        dl_upd_btn.click(dl_update, None, dl_msg)
        video.change(on_video, [video, out_root], [info, t_sl, y0, y1, use_logo, lx, ly, lw, lh]).then(
            show_regions, [video, t_sl, *cfg_in], img).then(
            load_table, [video, out_root, lang], tbl)
        for c in [t_sl, *cfg_in]:
            c.release(show_regions, [video, t_sl, *cfg_in], img) if isinstance(c, gr.Slider) else \
                c.change(show_regions, [video, t_sl, *cfg_in], img)
        ocr_btn.click(do_ocr, [video, out_root, step, force, *cfg_in], [ocr_tbl, ocr_msg]).then(
            load_table, [video, out_root, lang], tbl)
        tr_btn.click(do_translate, [video, out_root, lang, only_missing], [tbl, tr_msg])
        exp_btn.click(do_export, [video, out_root], exp_file)
        imp_btn.click(do_import, [video, out_root, lang, imp_file], [tbl, tr_msg])
        reload_btn.click(load_table, [video, out_root, lang], tbl)
        save_btn.click(save_table, [video, out_root, lang, tbl], edit_msg)
        tbl.select(preview_row, [video, out_root, lang, tbl, font, size, *cfg_in], prev)
        render_in = [video, out_root, lang, tbl, font, size, split_mode, split_sec, quality]
        test_btn.click(render_test, [*render_in, *cfg_in], [out_files, out_vid, out_msg])
        full_btn.click(render_full, [*render_in, *cfg_in], [out_files, out_vid, out_msg])
    return ui


if __name__ == "__main__":
    build().queue().launch(inbrowser=True)
