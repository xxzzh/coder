@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%~dp0"
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
"%PY%" scripts\package_dist.py
exit /b %ERRORLEVEL%
