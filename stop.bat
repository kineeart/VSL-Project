@echo off
chcp 65001 >nul
echo Dang dung tat ca servers...

:: Kill backend (uvicorn/python on port 8000)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Kill frontend (vite on port 3000)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo [OK] Da dung tat ca servers.
pause
