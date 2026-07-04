@echo off
REM Khoi dong he thong AI dem xe. Chay tu thu muc project.
cd /d "%~dp0"
set YOLO_CONFIG_DIR=%~dp0.cache\ultralytics
set HF_HOME=%~dp0.cache\huggingface
set FAST_PLATE_OCR_HUB_HOME=%~dp0.cache\fast_plate_ocr
set MPLCONFIGDIR=%~dp0.cache\matplotlib
"%~dp0venv\Scripts\python.exe" run.py
pause
