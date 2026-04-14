@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".env" (
  if exist ".env.example" copy /Y ".env.example" ".env" >nul
)

echo ============================================
echo   VSL Mobile Mini - Build APK (EAS)
echo ============================================
echo.

echo [1/3] Installing dependencies...
call npm install
if %errorlevel% neq 0 exit /b 1

echo.
echo [2/3] Ensure eas-cli is available...
call npx eas-cli --version
if %errorlevel% neq 0 exit /b 1

echo.
echo [3/3] Building preview APK in cloud...
call npx eas-cli build --platform android --profile preview
if %errorlevel% neq 0 exit /b 1

echo.
echo Build submitted. Check EAS dashboard for APK download link.
pause
