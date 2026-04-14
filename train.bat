@echo off
chcp 65001 >nul
title VSL - GPU Training

echo ============================================
echo   Vietnamese Sign Language Recognition
echo   Training Script (GPU)
echo ============================================
echo.

:: Try to find Python: .venv first, then system py
set PYTHON=
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else if exist "python.exe" (
    set PYTHON=python.exe
) else (
    for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON=%%i
)

if not defined PYTHON (
    echo [LOI] Khong tim thay Python!
    echo.
    echo Vui long cai dat Python 3.12+ hoac thay doi duong dan trong train.bat
    echo.
    echo Hoac cau hinh venv:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r backend/requirements.txt
    echo.
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON%
echo.

REM ===== MODEL CONFIGURATION =====
REM Change output model directory:
REM   - backend\models (default)
REM   - backend\models_15cls_run1 (15-class experiment)
REM   - etc.
set "VSL_MODEL_DIR=%CD%\backend\models"
REM ================================
echo [INFO] Model will be saved to: %VSL_MODEL_DIR%
echo.

REM Check torch CUDA
%PYTHON% -c "import torch; assert torch.cuda.is_available(), 'NO GPU'" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] Cai dat PyTorch GPU CUDA 12.1...
    %PYTHON% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo [1/3] PyTorch GPU da co san.
)

REM Check other deps
%PYTHON% -c "import fastapi, mediapipe" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/3] Cai dat thu vien...
    %PYTHON% -m pip install fastapi uvicorn mediapipe opencv-python-headless numpy openpyxl python-multipart scikit-learn --quiet
) else (
    echo [2/3] Thu vien da co san.
)

echo.
%PYTHON% -c "import torch; print(f'[OK] GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}  Torch: {torch.__version__}') if torch.cuda.is_available() else print('[CANH BAO] DANG DUNG CPU!')"

echo.
echo [3/3] Bat dau huan luyen...
echo ============================================
echo.

:: Ensure we're in the script's directory before cd backend
cd /d "%~dp0"

if not exist "backend" (
    echo [LOI] Khong tim thay thu muc backend!
    echo Vui long chay train.bat tu thu muc goc cua project.
    pause
    exit /b 1
)

cd backend
if not exist "models" mkdir "models"
if not exist "landmarks" mkdir "landmarks"
if not exist "custom_videos" mkdir "custom_videos"

set VSL_MODEL_DIR=%VSL_MODEL_DIR%
%PYTHON% train_gpu.py

echo.
echo ============================================
echo   Hoan tat! Nhan phim bat ky de dong.
echo ============================================
cd ..
pause
