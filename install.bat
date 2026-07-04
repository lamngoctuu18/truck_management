@echo off
setlocal enabledelayedexpansion
REM =====================================================================
REM   CAI DAT TU DONG - He thong AI dem xe cho dat/gach
REM   - Tao venv + cai toan bo thu vien vao thu muc project (o E)
REM   - Cache/temp huong ve o E de tranh o C day
REM =====================================================================
cd /d "%~dp0"

echo ============================================================
echo   CAI DAT HE THONG AI DEM XE CHO DAT/GACH
echo ============================================================
echo   Thu muc: %~dp0
echo.

REM ---- Huong cache/temp ve o E (khong dung o C dang day) ----
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "TMP=%~dp0.cache\tmp"
set "TEMP=%~dp0.cache\tmp"
set "TMPDIR=%~dp0.cache\tmp"
if not exist "%~dp0.cache\pip" mkdir "%~dp0.cache\pip"
if not exist "%~dp0.cache\tmp" mkdir "%~dp0.cache\tmp"

REM ---- 1. Kiem tra Python ----
echo [1/6] Kiem tra Python...
where python >nul 2>&1
if errorlevel 1 (
    echo    LOI: Khong tim thay Python. Hay cai Python 3.12 tu https://python.org
    echo    Nho tick "Add Python to PATH" khi cai.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo    Python !PYVER! OK
echo.

REM ---- 2. Tao virtual environment (venv) ----
echo [2/6] Tao moi truong ao (venv)...
if exist "%~dp0venv\Scripts\python.exe" (
    echo    venv da ton tai, dung lai venv cu.
) else (
    python -m venv "%~dp0venv"
    if errorlevel 1 (
        echo    LOI: Khong tao duoc venv.
        pause
        exit /b 1
    )
    echo    Da tao venv tai %~dp0venv
)
set "VPY=%~dp0venv\Scripts\python.exe"
echo.

REM ---- 3. Nang cap pip ----
echo [3/6] Nang cap pip...
"%VPY%" -m pip install --upgrade pip
echo.

REM ---- 4. Cai PyTorch CUDA (tu index rieng cua PyTorch) ----
echo [4/6] Cai PyTorch CUDA 12.1 (torch + torchvision)...
echo    (Tai ~2.5GB, co the mat vai phut, vui long doi...)
"%VPY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo    CANH BAO: Cai torch CUDA that bai. Thu cai ban CPU...
    "%VPY%" -m pip install torch==2.5.1 torchvision==0.20.1
)
echo.

REM ---- 5. Cai cac thu vien con lai ----
echo [5/6] Cai cac thu vien con lai (ultralytics, fastapi, fast-alpr...)...
if exist "%~dp0requirements.txt" (
    REM Bo torch/torchvision khoi requirements vi da cai o buoc 4
    "%VPY%" -c "import re; lines=[l for l in open('requirements.txt',encoding='utf-8') if not re.match(r'^(torch|torchvision)==', l.strip())]; open('.cache\\req_core.txt','w',encoding='utf-8').writelines(lines)"
    "%VPY%" -m pip install -r ".cache\req_core.txt"
) else (
    echo    Khong thay requirements.txt, cai truc tiep goi chinh...
    "%VPY%" -m pip install ultralytics fast-alpr onnxruntime fastapi "uvicorn[standard]" jinja2 python-multipart pymysql sqlalchemy python-dotenv websockets openpyxl
)
if errorlevel 1 (
    echo    LOI: Cai thu vien that bai. Kiem tra ket noi mang.
    pause
    exit /b 1
)
echo.

REM ---- 6. Kiem tra cai dat ----
echo [6/6] Kiem tra cai dat...
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
set "HF_HOME=%~dp0.cache\huggingface"
set "FAST_PLATE_OCR_HUB_HOME=%~dp0.cache\fast_plate_ocr"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
"%VPY%" -c "import torch,cv2,ultralytics,fastapi; print('   torch',torch.__version__,'| CUDA',torch.cuda.is_available()); print('   ultralytics',ultralytics.__version__,'| opencv',cv2.__version__,'| fastapi OK')"
if errorlevel 1 (
    echo    CANH BAO: Mot so thu vien chua import duoc, xem loi ben tren.
) else (
    echo.
    echo ============================================================
    echo   CAI DAT HOAN TAT!
    echo ============================================================
    echo   - Chay he thong: nhap dup file run.bat
    echo   - Dashboard: http://localhost:8000
    echo.
    echo   LUU Y: Can co PostgreSQL dang chay o localhost:5432
    echo          (user=postgres, mat khau=123, xem file .env)
    echo          He thong tu tao database 'vehicle_management' neu chua co.
    echo ============================================================
)
echo.
pause
