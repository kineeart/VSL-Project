@echo off
chcp 65001 >nul
title Vietnamese Sign Language Recognition

REM ====== MODEL CONFIGURATION ======
REM Change the model directory here to use different trained models:
REM   - backend\models (default)
REM   - backend\models_15cls_run1 (15-class experiment)
REM   - etc.
set "MODEL_DIR=backend\models_15cls_run1"
set "VSL_MODEL_DIR=%CD%\%MODEL_DIR%"
REM ===================================

echo ============================================
echo   Vietnamese Sign Language Recognition
echo   Khoi dong he thong...
echo ============================================
echo   Model used: %VSL_MODEL_DIR%
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Python. Vui long cai dat Python 3.10+
    pause
    exit /b 1
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo [0/4] Tao moi moi truong ao .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [LOI] Tao moi truong ao that bai.
        pause
        exit /b 1
    )
)

:: Check Node.js
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay Node.js. Vui long cai dat Node.js 18+
    pause
    exit /b 1
)

:: Check npm
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [LOI] Khong tim thay npm.
    pause
    exit /b 1
)

echo [OK] Python, Node.js, npm da san sang.
echo.

:: ---- Backend setup ----
echo [1/4] Cai dat thu vien Python (backend)...
cd backend
..\%PYTHON_EXE% -m pip install --upgrade pip --quiet
..\%PYTHON_EXE% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
..\%PYTHON_EXE% -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [LOI] Cai dat thu vien Python that bai.
    cd ..
    pause
    exit /b 1
)
echo [OK] Thu vien Python da cai xong.
cd ..
echo.

:: ---- Frontend setup ----
echo [2/4] Cai dat thu vien Node.js (frontend)...
cd frontend
call npm install --silent
if %errorlevel% neq 0 (
    echo [LOI] npm install that bai.
    cd ..
    pause
    exit /b 1
)
echo [OK] Thu vien Node.js da cai xong.
cd ..
echo.

:: ---- Create backend directories ----
echo [3/4] Tao thu muc can thiet...
if not exist "backend\models" mkdir "backend\models"
if not exist "backend\landmarks" mkdir "backend\landmarks"
if not exist "backend\custom_videos" mkdir "backend\custom_videos"
echo [OK] Thu muc da san sang.
echo.

:: ---- Start servers ----
echo [4/4] Khoi dong servers...
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo.
echo   Nhan Ctrl+C de dung tat ca.
echo ============================================
echo.

:: Start backend in background; keep window open if startup fails
start "VSL-Backend" cmd /k "cd backend && set VSL_MODEL_DIR=%VSL_MODEL_DIR% && ..\%PYTHON_EXE% -m uvicorn app:app --host 0.0.0.0 --port 8000"

:: Wait for backend to start
echo Dang cho backend khoi dong...
timeout /t 3 /nobreak >nul

:: Start frontend (foreground so closing this window stops everything)
cd frontend
call npm run dev
