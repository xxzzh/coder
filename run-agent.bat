@echo off
setlocal
chcp 65001 >nul

set "ROOT=%~dp0"
set "INGEST=%ROOT%scripts\ingest_knowledge_base.py"
set "QUERY=%ROOT%scripts\query_knowledge_base.py"
set "MANAGE=%ROOT%scripts\manage_knowledge_base.py"
set "SETUP=%ROOT%scripts\setup_agent.py"
set "HEALTH=%ROOT%scripts\healthcheck_agent.py"
set "WEB=%ROOT%scripts\web_agent.py"
set "TRANSLATION_SETUP=%ROOT%scripts\install_translation_model.py"
set "API_SETUP=%ROOT%scripts\configure_api.py"
set "EMBEDDING_SETUP=%ROOT%scripts\configure_embedding.py"
set "INSTALL_DEPS=%ROOT%scripts\install_project_dependencies.py"
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

if "%~1"=="" goto usage
if /I "%~1"=="install-deps" goto install_deps
if /I "%~1"=="setup" goto setup
if /I "%~1"=="verify" goto verify
if /I "%~1"=="healthcheck" goto verify
if /I "%~1"=="web" goto web
if /I "%~1"=="translation-setup" goto translation_setup
if /I "%~1"=="api-setup" goto api_setup
if /I "%~1"=="embedding-setup" goto embedding_setup
if /I "%~1"=="ingest" goto ingest
if /I "%~1"=="update" goto update
if /I "%~1"=="ask" goto ask
if /I "%~1"=="sources" goto sources
if /I "%~1"=="rebuild" goto rebuild
if /I "%~1"=="rebuild-strict" goto rebuild_strict
goto direct

:usage
echo Please enter a question or command.
echo.
echo Examples:
echo   .\run-agent.bat setup
echo   .\run-agent.bat install-deps
echo   .\run-agent.bat web
echo   .\run-agent.bat translation-setup
echo   .\run-agent.bat api-setup
echo   .\run-agent.bat embedding-setup
echo   .\run-agent.bat verify
echo   .\run-agent.bat "your question"
echo   .\run-agent.bat ingest
echo   .\run-agent.bat update
echo   .\run-agent.bat sources
echo   .\run-agent.bat rebuild
echo   .\run-agent.bat rebuild-strict
echo.
echo Direct questions query the local knowledge base first. If local evidence is insufficient, web search is used automatically and source_type is web_search.
exit /b 1

:install_deps
"%PY%" "%INSTALL_DEPS%"
exit /b %ERRORLEVEL%

:setup
"%PY%" "%SETUP%"
exit /b %ERRORLEVEL%

:verify
"%PY%" "%HEALTH%"
exit /b %ERRORLEVEL%

:web
shift
"%PY%" "%WEB%" %1 %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL%

:translation_setup
"%PY%" "%TRANSLATION_SETUP%"
exit /b %ERRORLEVEL%

:api_setup
"%PY%" "%API_SETUP%"
exit /b %ERRORLEVEL%

:embedding_setup
"%PY%" "%EMBEDDING_SETUP%"
exit /b %ERRORLEVEL%

:ingest
"%PY%" "%INGEST%" "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:update
"%PY%" "%MANAGE%" update "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:sources
"%PY%" "%MANAGE%" sources
exit /b %ERRORLEVEL%

:rebuild
"%PY%" "%MANAGE%" rebuild "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:rebuild_strict
"%PY%" "%MANAGE%" rebuild "%ROOT%knowledge_base\raw" --strict-extraction
exit /b %ERRORLEVEL%

:ask
if "%~2"=="" (
  echo Usage: .\run-agent.bat ask "question" [--no-web]
  exit /b 1
)
if /I "%~3"=="--no-web" (
  "%PY%" "%QUERY%" "%~2"
) else (
  "%PY%" "%QUERY%" "%~2" --web
)
exit /b %ERRORLEVEL%

:direct
"%PY%" "%QUERY%" "%~1" --web
exit /b %ERRORLEVEL%
