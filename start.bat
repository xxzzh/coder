@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "RAW=%ROOT%knowledge_base\raw"
set "RUNNER=%ROOT%run-agent.bat"
set "OCR_EVAL=%ROOT%scripts\evaluate_ocr_test_materials.py"
set "OCR_GEN=%ROOT%scripts\generate_ocr_test_materials.py"
set "PYTHONUTF8=1"

if not exist "%RAW%" mkdir "%RAW%"

:menu
cls
echo ================================================
echo  Local Knowledge Base Agent
echo ================================================
echo.
echo  1. First-time setup / configure API
echo  2. Start Web UI
echo  3. Open raw materials folder
echo  4. Strict rebuild index
echo  5. Incremental update index
echo  6. Health check
echo  7. Ask a question
echo  8. List indexed sources
echo  9. Run OCR test materials evaluation
echo 10. Generate OCR test materials
echo  0. Exit
echo.
set /p "CHOICE=Choose an option: "

if "%CHOICE%"=="1" goto setup
if "%CHOICE%"=="2" goto web
if "%CHOICE%"=="3" goto open_raw
if "%CHOICE%"=="4" goto rebuild_strict
if "%CHOICE%"=="5" goto update
if "%CHOICE%"=="6" goto verify
if "%CHOICE%"=="7" goto ask
if "%CHOICE%"=="8" goto sources
if "%CHOICE%"=="9" goto ocr_eval
if "%CHOICE%"=="10" goto ocr_generate
if "%CHOICE%"=="0" goto end

echo.
echo Invalid option.
pause
goto menu

:setup
call "%RUNNER%" setup
echo.
pause
goto menu

:web
echo.
echo Starting Web UI at http://127.0.0.1:8765/
echo Close this window or press Ctrl+C to stop the Web UI.
echo.
call "%RUNNER%" web --open
echo.
pause
goto menu

:open_raw
start "" "%RAW%"
echo.
echo Put .md, .docx, .pdf, and .xlsx files in this folder, then run option 3 or 4.
echo.
pause
goto menu

:rebuild_strict
call "%RUNNER%" rebuild-strict
echo.
echo If the result has requires_review greater than 0, inspect:
echo   knowledge_base\processed\extraction_report.jsonl
echo.
pause
goto menu

:update
call "%RUNNER%" update
echo.
pause
goto menu

:verify
call "%RUNNER%" verify
echo.
echo The agent is ready when the output contains: "ok": true
echo.
pause
goto menu

:ask
echo.
set /p "QUESTION=Enter your question: "
if "%QUESTION%"=="" (
  echo Question cannot be empty.
  pause
  goto menu
)
call "%RUNNER%" "%QUESTION%"
echo.
pause
goto menu

:sources
call "%RUNNER%" sources
echo.
pause
goto menu

:ocr_eval
if not exist "%OCR_EVAL%" (
  echo Missing OCR evaluation script: %OCR_EVAL%
  pause
  goto menu
)
python "%OCR_EVAL%"
echo.
pause
goto menu

:ocr_generate
if not exist "%OCR_GEN%" (
  echo Missing OCR material generator: %OCR_GEN%
  pause
  goto menu
)
python "%OCR_GEN%"
echo.
pause
goto menu

:end
endlocal
exit /b 0
