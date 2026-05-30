@echo off
setlocal

set "ROOT=%~dp0"
set "AGENT=%ROOT%examples\customer-support-agent\agent.py"
set "INGEST=%ROOT%scripts\ingest_knowledge_base.py"
set "QUERY=%ROOT%scripts\query_knowledge_base.py"
set "MANAGE=%ROOT%scripts\manage_knowledge_base.py"
set "PYTHONUTF8=1"

if "%~1"=="" goto usage
if /I "%~1"=="ingest" goto ingest
if /I "%~1"=="ask" goto ask
if /I "%~1"=="sources" goto sources
if /I "%~1"=="rebuild" goto rebuild
if /I "%~1"=="skills" goto skills
if /I "%~1"=="ticket" goto ticket
if /I "%~1"=="verbose" goto verbose
goto direct

:usage
echo Please enter a question or command.
echo.
echo Examples:
echo   .\run-agent.bat "your question"
echo   .\run-agent.bat ingest
echo   .\run-agent.bat sources
echo   .\run-agent.bat rebuild
echo.
echo Direct questions query the local knowledge base first. If local evidence is insufficient, web search is used automatically and source_type is web_search.
exit /b 1

:skills
python "%AGENT%" --list-skills
exit /b %ERRORLEVEL%

:ingest
python "%INGEST%" "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:sources
python "%MANAGE%" sources
exit /b %ERRORLEVEL%

:rebuild
python "%MANAGE%" rebuild "%ROOT%knowledge_base\raw"
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

:ticket
python "%AGENT%" --ticket "%~2"
exit /b %ERRORLEVEL%

:verbose
python "%AGENT%" --subject "%~2" --body "%~3" --verbose
exit /b %ERRORLEVEL%

:direct
python "%QUERY%" "%~1" --web
exit /b %ERRORLEVEL%
