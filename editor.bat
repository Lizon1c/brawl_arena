@echo off
rem brawl-arena map editor one-click launcher
rem double-click -> server starts -> browser opens http://127.0.0.1:8787
rem close this window to stop the editor server
cd /d "%~dp0"
start "" /min cmd /c "timeout /t 2 /nobreak >nul && start "" http://127.0.0.1:8787"
.venv\Scripts\python.exe editor.py
pause
