@echo off
cd /d "%~dp0"
chcp 65001 > nul
title AIC 2026 Multimodal Retrieval Engine

echo ========================================================
echo   AIC 2026 - KHỞI ĐỘNG HỆ THỐNG TÌM KIẾM MULTIMODAL
echo ========================================================
echo Thư mục hiện tại: %CD%
echo.

if not exist "venv\Scripts\python.exe" (
    echo [CẢNH BÁO] Chưa tìm thấy môi trường ảo venv!
    echo Đang tự động chạy cài đặt thư viện trước...
    call setup_environment.bat
)

if not exist "venv\Scripts\python.exe" (
    echo [LỖI] Không thể tạo môi trường venv! Vui lòng kiểm tra lại Python trên máy.
    pause
    exit /b 1
)

echo Đang khởi động Server FastAPI trên cổng 8000...
echo Tự động mở trình duyệt web...

start "" "http://127.0.0.1:8000"
.\venv\Scripts\python.exe server.py

if %errorlevel% neq 0 (
    echo.
    echo [LỖI] Server dừng lại với mã lỗi: %errorlevel%
)

pause
