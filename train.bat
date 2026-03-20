@echo off
chcp 65001 >nul
title VSL - GPU Training

echo ============================================
echo   Vietnamese Sign Language Recognition
echo   Training Script (GPU)
echo ============================================
echo.

set PYTHON=C:\Users\thang\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON%" (
    echo [LOI] Khong tim thay Python tai %PYTHON%
    pause
    exit /b 1
)

:: Check torch CUDA
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'NO GPU'" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Cai dat PyTorch GPU CUDA 12.1...
    %PYTHON% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo [1/3] PyTorch GPU da co san.
)

:: Check other deps
%PYTHON% -c "import fastapi, mediapipe" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] Cai dat thu vien...
    %PYTHON% -m pip install fastapi uvicorn mediapipe opencv-python-headless numpy openpyxl python-multipart scikit-learn --quiet
) else (
    echo [2/3] Thu vien da co san.
)

echo.
%PYTHON% -c "import torch; print(f'[OK] GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}  Torch: {torch.__version__}') if torch.cuda.is_available() else print('[CANH BAO] DANG DUNG CPU!')"

cd backend
if not exist "models" mkdir "models"
if not exist "landmarks" mkdir "landmarks"
if not exist "custom_videos" mkdir "custom_videos"

echo.
echo [3/3] Bat dau huan luyen...
echo ============================================
echo.
%PYTHON% train_gpu.py

echo.
echo ============================================
echo   Hoan tat! Nhan phim bat ky de dong.
echo ============================================
cd ..
pause
