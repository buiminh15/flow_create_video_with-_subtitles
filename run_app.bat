@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

rem Python đã cài sẵn các thư viện (gradio, rapidocr, opencv...) ở môi trường này
set "PY=C:\Users\ASUS\.virtualenvs\Real-Time-Voice-Cloning-Dnz8evWK\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Dang khoi dong app... (trinh duyet se tu mo, dong cua so nay de tat app)
"%PY%" app.py
if errorlevel 1 (
    echo.
    echo App bi loi, xem thong bao o tren.
    pause
)
