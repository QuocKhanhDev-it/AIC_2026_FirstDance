@echo off
cd /d "%~dp0"
chcp 65001 > nul
title AIC 2026 Environment Installer

echo ========================================================
echo   AIC 2026 - TỰ ĐỘNG CÀI ĐẶT MÔI TRƯỜNG (1-CLICK SETUP)
echo ========================================================
echo Thư mục cài đặt: %CD%
echo.

:: Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LỖI] Không tìm thấy Python trên máy tính này!
    echo Vui lòng cài đặt Python 3.10 hoặc 3.11 từ https://www.python.org/
    echo Lưu ý: Nhớ tick chọn "Add Python to PATH" khi cài đặt.
    echo.
    pause
    exit /b 1
)

echo [1/4] Đang tạo môi trường ảo (venv)...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    echo  -> Đã tạo thư mục venv thành công.
) else (
    echo  -> Môi trường venv đã sẵn sàng.
)

echo.
echo [2/4] Đang nâng cấp pip...
.\venv\Scripts\python.exe -m pip install --upgrade pip -q

echo.
echo [3/4] Đang cài đặt các thư viện cần thiết (requirements.txt)...
echo Quá trình này mất khoảng 2-3 phút tuỳ tốc độ mạng...
.\venv\Scripts\pip.exe install -r requirements.txt

echo.
echo [4/4] Kiểm tra file cấu hình .env...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env > nul
        echo  -> Đã tạo file .env từ .env.example.
    ) else (
        echo GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE > .env
        echo GEMINI_MODEL=gemini-flash-latest >> .env
        echo  -> Đã khởi tạo file .env mẫu.
    )
) else (
    echo  -> File .env đã sẵn sàng.
)

echo.
echo ========================================================
echo   ✅ CÀI ĐẶT THÀNH CÔNG 100%!
echo   Bây giờ bạn chỉ cần chạy file 'start_server.bat' để mở Web!
echo ========================================================
echo.
pause
