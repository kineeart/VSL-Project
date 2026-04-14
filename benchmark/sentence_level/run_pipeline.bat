@echo off
chcp 65001 >nul
setlocal
setlocal EnableDelayedExpansion

cd /d "%~dp0\..\.."

set "DATASET=benchmark\sentence_level\data\sentence_dataset.synthetic.small.json"
set "VALIDATION_OUT=benchmark\sentence_level\data\sentence_dataset.synthetic.small.validation.json"
set "SPLIT_OUT=benchmark\sentence_level\data\sentence_dataset.synthetic.small.split.json"
set "SPLIT_REPORT=benchmark\sentence_level\data\sentence_dataset.synthetic.small.split.report.json"
set "PRED_OUT=benchmark\sentence_level\data\sentence_predictions.synthetic.small.json"
set "EVAL_OUT=benchmark\sentence_level\data\sentence_eval.synthetic.small.json"
set "BACKEND=http://127.0.0.1:8000"
set "SPLIT=test"
set "GROUP_BY=signer"
set "ALLOW_FALLBACK_SPLIT=true"

if not "%~1"=="" set "DATASET=%~1"
if not "%~2"=="" set "BACKEND=%~2"


echo ============================================================
echo   VSL Sentence-Level Pipeline (Robust Research)
echo ============================================================
echo Dataset : %DATASET%
echo Backend : %BACKEND%
echo Split   : %SPLIT%
echo GroupBy : %GROUP_BY%
echo ============================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Python not found.
  exit /b 1
)

if not exist "%DATASET%" (
  echo [ERROR] Dataset not found: %DATASET%
  exit /b 1
)

echo [1/5] Validate dataset...
python benchmark/sentence_level/scripts/validate_sentence_dataset.py --input "%DATASET%" --out "%VALIDATION_OUT%"
if %errorlevel% geq 2 (
  echo [ERROR] Validation has ERROR-level issues. Stop pipeline.
  exit /b 2
)

echo.
echo [2/5] Split dataset (signer-safe)...
python benchmark/sentence_level/scripts/split_sentence_dataset.py --input "%DATASET%" --output "%SPLIT_OUT%" --report "%SPLIT_REPORT%" --group-by !GROUP_BY! --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
if %errorlevel% neq 0 (
  echo [ERROR] Split failed.
  exit /b 3
)

python -c "import json; r=json.load(open(r'%SPLIT_REPORT%','r',encoding='utf-8')); es=r.get('empty_splits',[]); print(f'empty_splits={es}'); exit(0 if 'test' not in es else 7)"
if %errorlevel% neq 0 (
  if /i "%ALLOW_FALLBACK_SPLIT%"=="true" (
    echo [WARN] Test split empty with GROUP_BY=%GROUP_BY%. Fallback to sample-level split for runnable pipeline.
    set "GROUP_BY=sample"
    python benchmark/sentence_level/scripts/split_sentence_dataset.py --input "%DATASET%" --output "%SPLIT_OUT%" --report "%SPLIT_REPORT%" --group-by !GROUP_BY! --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
    if !errorlevel! neq 0 (
      echo [ERROR] Fallback split failed.
      exit /b 3
    )
  ) else (
    echo [ERROR] Test split empty and fallback disabled.
    exit /b 3
  )
)

echo.
echo [3/5] Generate predictions from backend API...
python benchmark/sentence_level/scripts/generate_sentence_predictions_from_backend.py --dataset "%SPLIT_OUT%" --output "%PRED_OUT%" --backend "%BACKEND%" --split %SPLIT%
if %errorlevel% neq 0 (
  echo [ERROR] Prediction generation failed. Is backend running at %BACKEND% ?
  exit /b 4
)

echo.
echo [4/5] Evaluate with breakdown (dialect + signer)...
python benchmark/sentence_level/scripts/evaluate_sentence_benchmark.py --gt "%SPLIT_OUT%" --pred "%PRED_OUT%" --split %SPLIT% --breakdown --out "%EVAL_OUT%"
if %errorlevel% neq 0 (
  echo [ERROR] Evaluate failed.
  exit /b 5
)

echo.
echo [5/5] Quick quality gate...
python -c "import json; r=json.load(open(r'%EVAL_OUT%','r',encoding='utf-8')); n=r.get('num_evaluated',0); wer=r.get('wer',1.0); miss=r.get('num_missing_predictions',99999); print(f'num_evaluated={n}, wer={wer:.4f}, missing={miss}'); exit(0 if (n>0 and miss==0) else 6)"
if %errorlevel% neq 0 (
  echo [WARN] Pipeline ran, but quality gate NOT passed. Check outputs below.
  echo   - %VALIDATION_OUT%
  echo   - %SPLIT_REPORT%
  echo   - %PRED_OUT%
  echo   - %EVAL_OUT%
  exit /b 6
)

echo.
echo [SUCCESS] Pipeline completed and passed minimum quality gate.
if /i "%GROUP_BY%"=="sample" (
  echo [WARN] Running with GROUP_BY=sample (not signer-safe). Use only for quick debug, not final research claims.
)
echo Outputs:
echo   - %VALIDATION_OUT%
echo   - %SPLIT_REPORT%
echo   - %PRED_OUT%
echo   - %EVAL_OUT%
exit /b 0
