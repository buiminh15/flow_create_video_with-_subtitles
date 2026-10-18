"""CLI: OCR phụ đề cứng -> dịch -> dựng video. Giao diện web: python app.py

Dùng:  python translate_video.py clip.mp4 --lang vi --split 20
Kết quả nằm trong output/<tên video>/ (có cache; sửa translated.<lang>.json rồi chạy lại để áp dụng).
"""
import argparse
import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--split", type=float, default=20, help="độ dài mỗi clip (giây), 0 = không cắt")
    ap.add_argument("--split-mode", choices=["sentence", "fixed"], default="sentence")
    ap.add_argument("--step", type=float, default=0.5, help="chu kỳ lấy mẫu OCR (giây)")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--quality", choices=list(core.QUALITY), default="cân bằng")
    ap.add_argument("--out", default="output")
    a = ap.parse_args()

    pd = core.project_dir(a.video, a.out)
    cfg = core.load_cfg(pd)
    core.save_cfg(pd, cfg)
    show = lambda f, m: print(f"[{f:4.0%}] {m}", flush=True)

    segs = core.run_ocr(a.video, pd, cfg, step=a.step, cb=show)
    tp = core.tr_path(pd, a.lang)
    segs = core.ensure_ids(core.read_json(tp) or segs)
    miss = core.translate_segments(segs, a.lang, cb=show)
    core.write_json(tp, segs)
    if miss:
        raise SystemExit(f"{miss} đoạn chưa dịch được; chạy lại sau ít phút hoặc sửa {tp}")
    files = core.render(a.video, pd, segs, a.lang, cfg, a.font, 0, a.split_mode if a.split else "none",
                        a.split, a.quality, cb=show)
    print("Xong:", *files, sep="\n  ")


if __name__ == "__main__":
    main()
