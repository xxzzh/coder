@echo off
setlocal

set "ROOT=%~dp0"
set "AGENT=%ROOT%examples\customer-support-agent\agent.py"
set "INGEST=%ROOT%scripts\ingest_knowledge_base.py"
set "QUERY=%ROOT%scripts\query_knowledge_base.py"
set "PYTHONUTF8=1"

if "%~1"=="" goto interactive
if /I "%~1"=="ingest" goto ingest
if /I "%~1"=="ask" goto ask
if /I "%~1"=="skills" goto skills
if /I "%~1"=="ticket" goto ticket
if /I "%~1"=="verbose" goto verbose
goto direct

:interactive
set /p SUBJECT=Ticket subject: 
set /p BODY=Ticket body: 
python "%AGENT%" --subject "%SUBJECT%" --body "%BODY%"
exit /b %ERRORLEVEL%

:skills
python "%AGENT%" --list-skills
exit /b %ERRORLEVEL%

:ingest
python "%INGEST%" "%ROOT%knowledge_base\raw"
exit /b %ERRORLEVEL%

:ask
if "%~2"=="" (
  echo Usage: .\run-agent.bat ask "question" [--web]
  exit /b 1
)
if /I "%~3"=="--web" (
  python "%QUERY%" "%~2" --web
) else (
  python "%QUERY%" "%~2"
)
exit /b %ERRORLEVEL%

:ticket
python "%AGENT%" --ticket "%~2"
exit /b %ERRORLEVEL%

:verbose
python "%AGENT%" --subject "%~2" --body "%~3" --verbose
exit /b %ERRORLEVEL%

:direct
python "%AGENT%" --subject "%~1" --body "%~2"
exit /b %ERRORLEVEL%
