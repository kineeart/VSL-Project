@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0\..\.."

set "DATASET=benchmark\continuous\data\continuous_dataset.real.json"
set "PRED_OUT=benchmark\continuous\data\continuous_predictions.real.json"
set "EVAL_OUT=benchmark\continuous\data\continuous_eval.real.json"
set "BACKEND=http://127.0.0.1:8000"
set "SPLIT=test"
set "STRIDE=3"

if not "%~1"=="" set "DATASET=%~1"
if not "%~2"=="" set "BACKEND=%~2"

echo ============================================================
echo   VSL Continuous Real-Data Pipeline
echo ============================================================
echo Dataset : %DATASET%
echo Backend : %BACKEND%
echo Split   : %SPLIT%
echo Stride  : %STRIDE%
echo ============================================================
echo.

if not exist "%DATASET%" (
  echo [ERROR] Dataset not found: %DATASET%
  exit /b 1
)

echo [1/3] Generate predictions from continuous endpoint...
python benchmark/continuous/scripts/generate_continuous_predictions_from_backend.py --dataset "%DATASET%" --output "%PRED_OUT%" --backend "%BACKEND%" --split %SPLIT% --stride %STRIDE%
if %errorlevel% neq 0 (
  echo [ERROR] Prediction generation failed.
  exit /b 2
)

echo.
echo [2/3] Evaluate CER/WER/Exact-Match...
python benchmark/continuous/scripts/evaluate_continuous_benchmark.py --gt "%DATASET%" --pred "%PRED_OUT%" --split %SPLIT% --breakdown --out "%EVAL_OUT%"
if %errorlevel% neq 0 (
  echo [ERROR] Evaluation failed.
  exit /b 3
)

echo.
echo [3/3] Done. Outputs:
echo   - %PRED_OUT%
echo   - %EVAL_OUT%
exit /b 0
