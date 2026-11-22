@echo off
setlocal
set "DASH_DIR=%~dp0"
set "DASH_ENV=%LOCALAPPDATA%\MsPacManDashboard\python312"
if not exist "%DASH_ENV%\Scripts\python.exe" (
    py -3.12 -m venv "%DASH_ENV%"
    if errorlevel 1 goto failed
)
"%DASH_ENV%\Scripts\python.exe" -c "import ale_py, gymnasium, imageio_ffmpeg, PIL" >nul 2>&1
if errorlevel 1 (
    "%DASH_ENV%\Scripts\python.exe" -m pip install -r "%DASH_DIR%requirements.txt"
    if errorlevel 1 goto failed
)
"%DASH_ENV%\Scripts\python.exe" "%DASH_DIR%server.py" --open %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Kunne ikke starte MsPacMan Lab. Se feilmeldingen ovenfor.
pause
exit /b 1
