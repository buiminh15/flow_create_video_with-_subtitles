# Video Bilibili – OCR & dịch phụ đề cứng

Tải video (Bilibili...), nhận diện chữ cứng trên hình bằng OCR (RapidOCR), dịch sang tiếng Việt (hoặc ngôn ngữ khác) và render lại video có phụ đề.

## Yêu cầu

- Python 3.10+ (đã dùng thử với 3.12)
- **ffmpeg** có trong `PATH` (không cài được bằng pip). Kiểm tra: `ffmpeg -version`
- Kết nối Internet (dịch bằng Google Translate / MyMemory qua `deep-translator`)

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

**Giao diện web (Gradio):**

```powershell
python app.py
```

Hoặc bấm đúp [run_app.bat](run_app.bat). Trình duyệt tự mở.

**Dòng lệnh:**

```powershell
python translate_video.py video.mp4 --lang vi --split 20 --quality "cân bằng" --out output
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--lang` | `vi` | Ngôn ngữ đích |
| `--split` | `20` | Độ dài mỗi clip (giây), `0` = không cắt |
| `--split-mode` | `sentence` | `sentence` hoặc `fixed` |
| `--step` | `0.5` | Chu kỳ lấy mẫu OCR (giây) |
| `--font` | `Arial` | Font phụ đề |
| `--quality` | `cân bằng` | Mức chất lượng render |
| `--out` | `output` | Thư mục kết quả |

## Cấu trúc thư mục

- `downloads/` – video đã tải (bị git bỏ qua)
- `output/<tên video>/` – kết quả từng video: `ocr.json`, `for_translate.json`, `translated.vi.json`, `sub.vi.ass`, `clips/` (bị git bỏ qua)

## Lưu ý

- `run_app.bat` ưu tiên dùng một virtualenv cố định trên máy tác giả; nếu không tồn tại sẽ dùng `python` trong PATH. Hãy sửa biến `PY` nếu cần.
- Lần chạy đầu `rapidocr-onnxruntime` có thể mất thời gian khởi tạo model.
- Render video tốn CPU/thời gian; video dài nên để `--split` cắt thành clip.
- Chữ Trung trên video nền phức tạp có thể OCR sai – kiểm tra `for_translate.json` trước khi dịch nếu cần chính xác.
- Chỉ tải/dịch nội dung bạn có quyền sử dụng.
