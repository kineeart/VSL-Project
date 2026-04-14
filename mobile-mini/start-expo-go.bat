@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   VSL Mobile Mini - Expo Go (LAN)
echo ============================================
echo.

if not exist "node_modules" (
  echo [1/3] Installing dependencies...
  call npm install
  if %errorlevel% neq 0 exit /b 1
) else (
  echo [1/3] Dependencies already installed.
)

echo.
echo [2/3] Check backend host in .env
echo       Example: EXPO_PUBLIC_BACKEND_HOST=192.168.x.x:8000
echo.

echo [3/3] Starting Expo Dev Server (LAN)...
echo       Scan QR code with Expo Go on your phone.
echo.
call npx expo start --lan
