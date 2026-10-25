"""Transcribe (audio/video -> text) và Text-to-Speech dùng Gemini API (Google AI Studio).

Cần API key lấy tại https://aistudio.google.com/apikey — dán vào ô "API key" trên giao diện,
hoặc đặt biến môi trường GEMINI_API_KEY / GOOGLE_API_KEY.
"""
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

TRANSCRIBE_MODEL = "gemini-flash-latest"
TTS_MODEL = "gemini-3.8-flash-tts"

# Giọng đọc có sẵn của Gemini TTS (tham khảo tài liệu aistudio.google.com/docs/transcribe)
VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

# Gợi ý phong cách đọc (có thể sửa/tự nhập tuỳ ý — Gemini TTS hiểu được mô tả tự nhiên)
STYLE_PRESETS = [
    "Đọc tự nhiên, rõ ràng",
    "Đọc chậm, nhẹ nhàng, thong thả",
    "Vui vẻ, hào hứng, tràn đầy năng lượng",
    "Nghiêm túc, trang trọng như đọc tin tức",
    "Thầm thì, ấm áp, thân mật",
    "Kể chuyện, giàu cảm xúc, có cao độ lên xuống",
    "Nhanh, dồn dập, gấp gáp",
    "Buồn, trầm lắng, chậm rãi",
    "Giọng quảng cáo, thuyết phục, cuốn hút",
    "Giọng trẻ con, hồn nhiên, tinh nghịch",
]

Progress = lambda frac, msg="": None


def _with_retry(fn, tries=3, wait=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or "UNAVAILABLE" not in str(e) and "503" not in str(e):
                raise
            time.sleep(wait)


def _client(api_key):
    key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Chưa có API key. Lấy tại https://aistudio.google.com/apikey")
    return genai.Client(api_key=key)


SEG_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "start": types.Schema(type=types.Type.NUMBER),
            "end": types.Schema(type=types.Type.NUMBER),
            "text": types.Schema(type=types.Type.STRING),
        },
        required=["start", "end", "text"],
    ),
)


def _ascii_safe_copy(path):
    """Gemini SDK gửi tên file gốc trong HTTP header (chỉ ascii) khi upload resumable.
    Tên có dấu tiếng Việt/Trung sẽ làm httpx lỗi UnicodeEncodeError -> copy sang tên ascii tạm."""
    path = Path(path)
    try:
        path.name.encode("ascii")
        return path, None
    except UnicodeEncodeError:
        tmp = Path(tempfile.gettempdir()) / f"gemini_upload_{int(time.time() * 1000)}{path.suffix}"
        shutil.copyfile(path, tmp)
        return tmp, tmp


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".wmv", ".m4v"}


def _extract_audio(path):
    """Bóc tách audio ra file nén nhỏ (mono 16kHz) bằng ffmpeg -> upload lên Gemini nhanh/rẻ hơn upload
    cả video (Gemini tính token theo cả khung hình nếu gửi video, không cần cho việc transcribe)."""
    tmp = Path(tempfile.gettempdir()) / f"gemini_audio_{int(time.time() * 1000)}.m4a"
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn",
                        "-ac", "1", "-ar", "16000", "-b:a", "32k", str(tmp)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not tmp.exists():
        raise RuntimeError("Lỗi bóc tách audio: " + r.stderr[-300:])
    return tmp


def transcribe(path, api_key, language_hint="", cb=Progress):
    """Transcribe file âm thanh/video bằng Gemini. Trả về list segment [{id,start,end,text}]."""
    client = _client(api_key)
    is_video = Path(path).suffix.lower() in VIDEO_EXTS
    tmp_audio = None
    if is_video:
        cb(0.02, "Đang bóc tách audio từ video...")
        tmp_audio = _extract_audio(path)
        upload_path, tmp_to_clean = tmp_audio, tmp_audio
    else:
        upload_path, tmp_to_clean = _ascii_safe_copy(path)
    cb(0.1, "Đang tải lên Gemini...")
    mime = mimetypes.guess_type(str(upload_path))[0] or "audio/mp4"
    try:
        uploaded = client.files.upload(file=str(upload_path), config={"mime_type": mime})
    finally:
        if tmp_to_clean:
            tmp_to_clean.unlink(missing_ok=True)
    while uploaded.state and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and uploaded.state.name == "FAILED":
        raise RuntimeError("Gemini xử lý file lỗi.")

    cb(0.4, "Đang nhận diện giọng nói...")
    hint = f" Ngôn ngữ trong audio: {language_hint}." if language_hint else ""
    prompt = (
        "Transcribe toàn bộ audio này thành các đoạn câu, kèm thời điểm bắt đầu/kết thúc (giây, số thực)."
        + hint
        + " Trả về đúng JSON là một mảng các object {start, end, text}, không thêm chữ nào khác."
    )
    resp = _with_retry(lambda: client.models.generate_content(
        model=TRANSCRIBE_MODEL,
        contents=[uploaded, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SEG_SCHEMA,
        ),
    ))
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    data = json.loads(resp.text)
    segs = [{"id": i + 1, "start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
             "text": str(s["text"]).strip(), "tr": ""} for i, s in enumerate(data) if str(s.get("text", "")).strip()]
    cb(1.0, f"Transcribe xong: {len(segs)} đoạn")
    return segs


TR_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "id": types.Schema(type=types.Type.INTEGER),
            "tr": types.Schema(type=types.Type.STRING),
        },
        required=["id", "tr"],
    ),
)
LANG_NAMES = {"vi": "Vietnamese", "en": "English", "id": "Indonesian", "th": "Thai"}


def translate_segments(segs, target_lang, only_missing=True, cb=Progress, api_key=""):
    """Dịch segs (list {id,text,tr,...}) sang target_lang bằng Gemini. Sửa 'tr' ngay trên segs (in-place).
    Không giới hạn ngôn ngữ nguồn (tự nhận diện), phù hợp phụ đề tiếng Nhật/Hàn/Anh/... -> Việt."""
    client = _client(api_key)
    todo = [s for s in segs if not (only_missing and s.get("tr"))]
    if not todo:
        cb(1.0, "Không có đoạn nào cần dịch")
        return
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    chunks, cur, size = [], [], 0
    for s in todo:
        if cur and (size + len(s["text"]) > 6000 or len(cur) >= 100):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(s)
        size += len(s["text"]) + 1
    if cur:
        chunks.append(cur)

    for ci, chunk in enumerate(chunks):
        cb(ci / len(chunks), f"Đang dịch khối {ci + 1}/{len(chunks)}")
        payload = json.dumps([{"id": s["id"], "text": s["text"]} for s in chunk], ensure_ascii=False)
        prompt = (
            f"Dịch trường 'text' của mỗi object sau sang {lang_name}, giữ nguyên nghĩa, văn nói tự nhiên, "
            f"ngắn gọn phù hợp phụ đề. Trả về JSON là mảng các object {{id, tr}} theo đúng id đầu vào, "
            f"không thêm chữ nào khác.\n{payload}"
        )
        resp = _with_retry(lambda: client.models.generate_content(
            model=TRANSCRIBE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TR_SCHEMA),
        ))
        by_id = {int(r["id"]): str(r["tr"]).strip() for r in json.loads(resp.text)}
        for s in chunk:
            if s["id"] in by_id:
                s["tr"] = by_id[s["id"]]
    cb(1.0, "Dịch xong")


def _save_wav(path, pcm_bytes, rate=24000, channels=1, sample_width=2):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


def _pcm_tts(client, text, voice, style):
    content = f"{style.strip()}: {text.strip()}" if style.strip() else text.strip()
    resp = _with_retry(lambda: client.models.generate_content(
        model=TTS_MODEL,
        contents=content,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    ))
    return resp.candidates[0].content.parts[0].inline_data.data


def text_to_speech(text, api_key, voice="Kore", out_path="output/tts.wav", style="", cb=Progress):
    """Đọc văn bản thành giọng nói bằng Gemini TTS. Trả về đường dẫn file .wav."""
    if not text or not text.strip():
        raise RuntimeError("Chưa có nội dung để đọc.")
    client = _client(api_key)
    cb(0.2, "Đang tạo giọng nói...")
    pcm = _pcm_tts(client, text, voice, style)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_wav(out_path, pcm)
    cb(1.0, "Xong")
    return str(out_path)


def dub_track(segs, api_key, voice, style, total_duration, out_path, cb=Progress):
    """Tạo 1 file audio dài bằng total_duration (giây), mỗi đoạn giọng đọc (tr hoặc text) được chèn
    đúng tại thời điểm 'start' của đoạn đó (khoảng trống giữa các đoạn là im lặng). Dùng để ghép thế
    audio gốc khi xuất video (lồng tiếng theo đúng timing phụ đề). Trả về đường dẫn file .wav."""
    client = _client(api_key)
    rate = 24000
    total_samples = max(1, int(total_duration * rate))
    buf = bytearray(total_samples * 2)  # im lặng (16-bit mono)
    todo = [s for s in segs if (s.get("tr") or s.get("text") or "").strip()]
    for i, s in enumerate(todo):
        cb(i / max(1, len(todo)), f"Đang tạo giọng đoạn {i + 1}/{len(todo)}")
        text = (s.get("tr") or s["text"]).strip()
        pcm = _pcm_tts(client, text, voice, style)
        start_byte = max(0, int(s["start"] * rate)) * 2
        n = min(len(pcm), max(0, len(buf) - start_byte))
        if n > 0:
            buf[start_byte:start_byte + n] = pcm[:n]
    _save_wav(out_path, bytes(buf), rate=rate)
    cb(1.0, "Xong giọng đọc")
    return str(out_path)
