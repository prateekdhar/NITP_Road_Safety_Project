@echo off
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
if exist ".venv\Scripts\python.exe" (
 ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8503
) else if exist "..\Project\.venv\Scripts\python.exe" (
 "..\Project\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8503
) else (
 echo See START_HERE.md to set up Python.
 pause
)
