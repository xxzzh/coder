@echo off
setlocal

set "ROOT=%~dp0"
set "INGEST=%ROOT%scripts\ingest_knowledge_base.py"
set "QUERY=%ROOT%scripts\query_knowledge_base.py"
set "MANAGE=%ROOT%scripts\manage_knowledge_base.py"
set "SETUP=%ROOT%scripts\setup_agent.py"
set "HEALTH=%ROOT%scripts\healthcheck_agent.py"
set "PYTHONUTF8=1"

if "%~1"=="" goto usage
if /I "%~1"=="setup" goto setup
if /I "%~1"=="verify" goto verify
if /I "%~1"=="healthcheck" goto verify
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

:setup
python "%SETUP%"
exit /b %ERRORLEVEL%

:verify
python "%HEALTH%"
exit /b %ERRORLEVEL%

:ingest
python "%INGEST%" "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:update
python "%MANAGE%" update "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:sources
python "%MANAGE%" sources
exit /b %ERRORLEVEL%

:rebuild
python "%MANAGE%" rebuild "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:rebuild_strict
python "%MANAGE%" rebuild "%ROOT%knowledge_base\raw" --strict-extraction
exit /b %ERRORLEVEL%

:ask
if "%~2"=="" (
  echo Usage: .\run-agent.bat ask "question" [--no-web]
  exit /b 1
)
if /I "%~3"=="--no-web" (
  python "%QUERY%" "%~2"
) else (
  python "%QUERY%" "%~2" --web
)
exit /b %ERRORLEVEL%

:direct
python "%QUERY%" "%~1" --web
exit /b %ERRORLEVEL%
