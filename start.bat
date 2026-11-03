@echo off
setlocal EnableExtensions
chcp 65001 >nul
rem Run from this install's folder, so a start from anywhere else uses this knowledge base.
cd /d "%~dp0"

set "ROOT=%~dp0"
set "RAW=%ROOT%knowledge_base\raw"
set "RUNNER=%ROOT%run-agent.bat"
set "OCR_EVAL=%ROOT%scripts\evaluate_ocr_test_materials.py"
set "OCR_GEN=%ROOT%scripts\generate_ocr_test_materials.py"
set "MENU_TEXT=%ROOT%scripts\start_menu.py"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "LKA_PROJECT_ROOT=%ROOT%"
set "LKA_PROJECT_CACHE_DIR=%ROOT%tools\cache"
set "PIP_CACHE_DIR=%ROOT%tools\cache\pip"
set "XDG_CACHE_HOME=%ROOT%tools\cache\xdg-cache"
set "XDG_DATA_HOME=%ROOT%tools\cache\xdg-data"
set "HF_HOME=%ROOT%tools\cache\huggingface"
set "HF_HUB_CACHE=%ROOT%tools\cache\huggingface\hub"
set "TRANSFORMERS_CACHE=%ROOT%tools\cache\huggingface\transformers"
set "TORCH_HOME=%ROOT%tools\cache\torch"
set "PLAYWRIGHT_BROWSERS_PATH=%ROOT%tools\cache\playwright-browsers"
set "NPM_CONFIG_CACHE=%ROOT%tools\cache\npm"
set "ARGOS_PACKAGES_DIR=%ROOT%tools\cache\argos"
set "ARGOS_PACKAGE_DIR=%ROOT%tools\cache\argos"
if not exist "%ROOT%tools\cache" mkdir "%ROOT%tools\cache"
if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PY=%ROOT%.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

if not exist "%RAW%" mkdir "%RAW%"

:menu
cls
set "CHOICE="
rem The menu text is Chinese, printed by Python; without a working Python, the English menu below.
"%PY%" "%MENU_TEXT%" menu 2>nul || goto menu_ascii
set /p "CHOICE="
goto choose

:menu_ascii
echo ================================================
echo  Local Knowledge Base Agent
echo  (Python 3.10+ was not found: install it with "Add python.exe to PATH")
echo ================================================
echo.
echo  First install, in this order
echo   1. Install dependencies
echo   2. Create the configuration
echo   3. Install the local embedding model (about 2.2 GB)
echo   4. Configure the answer API (optional)
echo  Daily use
echo   5. Start the Web UI (upload documents, ask questions)
echo   6. Check the installation and documents
echo  15. Share on the local network (set an access password in Settings first)
echo  Without uploading in the Web UI
echo   7. Open the documents folder
echo   8. Process new and changed documents
echo   9. Process all documents again
echo  Other
echo  10. Ask a question here
echo  11. List processed documents
echo  12. Install the offline English-to-Chinese model
echo  13. Evaluate OCR test materials
echo  14. Generate OCR test materials
echo   0. Exit
echo.
set /p "CHOICE=Choose an option: "

:choose
if "%CHOICE%"=="1" goto install_deps
if "%CHOICE%"=="2" goto setup
if "%CHOICE%"=="3" goto embedding_setup
if "%CHOICE%"=="4" goto api_setup
if "%CHOICE%"=="5" goto web
if "%CHOICE%"=="6" goto verify
if "%CHOICE%"=="7" goto open_raw
if "%CHOICE%"=="8" goto update
if "%CHOICE%"=="9" goto rebuild_strict
if "%CHOICE%"=="10" goto ask
if "%CHOICE%"=="11" goto sources
if "%CHOICE%"=="12" goto translation_setup
if "%CHOICE%"=="13" goto ocr_eval
if "%CHOICE%"=="14" goto ocr_generate
if "%CHOICE%"=="15" goto web_lan
if "%CHOICE%"=="0" goto end

echo.
call :say invalid
pause
goto menu

:setup
call "%RUNNER%" setup
echo.
pause
goto menu

:web
echo.
call :say web
echo.
call "%RUNNER%" web --open
echo.
pause
goto menu

:web_lan
echo.
call :say web_lan
echo.
call "%RUNNER%" web --host 0.0.0.0 --open
echo.
pause
goto menu

:open_raw
start "" "%RAW%"
echo.
call :say open_raw
echo.
pause
goto menu

:rebuild_strict
call "%RUNNER%" rebuild-strict
echo.
call :say rebuild_strict
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
call :say verify
echo.
pause
goto menu

:ask
echo.
call :say ask_prompt
set "QUESTION="
set /p "QUESTION="
if "%QUESTION%"=="" (
  call :say ask_empty
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
  call :say ocr_eval_missing
  pause
  goto menu
)
"%PY%" "%OCR_EVAL%"
echo.
pause
goto menu

:ocr_generate
if not exist "%OCR_GEN%" (
  call :say ocr_gen_missing
  pause
  goto menu
)
"%PY%" "%OCR_GEN%"
echo.
pause
goto menu

:translation_setup
call "%RUNNER%" translation-setup
echo.
pause
goto menu

:api_setup
call "%RUNNER%" api-setup
echo.
pause
goto menu

:install_deps
call "%RUNNER%" install-deps
echo.
pause
goto menu

:embedding_setup
call "%RUNNER%" embedding-setup
echo.
pause
goto menu

:say
rem One of the Chinese messages in scripts\start_menu.py.
"%PY%" "%MENU_TEXT%" %1 2>nul
exit /b 0

:end
endlocal
exit /b 0
