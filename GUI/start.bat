@echo off
title YOLO Detection System - Gradio Version
color 0A

:: ============================================
:: YOLO Detection System - Windows Startup Script
:: ============================================

setlocal EnableDelayedExpansion

:: Set working directory to script location
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo.
echo ============================================================
echo.
echo    YOLO Object Detection System - Gradio Version
echo.
echo    Deep Learning Based Intelligent Detection Platform
echo.
echo ============================================================
echo.

:: ============================================
:: Step 1: Check Python Environment
:: ============================================
echo [1/5] Checking Python environment...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python environment not detected!
    echo.
    echo Please install Python:
    echo    1. Visit https://www.python.org/downloads/
    echo    2. Download and install Python 3.8 or higher
    echo    3. Check "Add Python to PATH" during installation
    echo    4. Restart this script
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo Python installed: %PYTHON_VERSION%
echo.

:: ============================================
:: Step 2: Check Project Directory
:: ============================================
echo [2/5] Checking project directory...
echo Current directory: %SCRIPT_DIR%

:: Check if key files exist
if not exist "run.py" (
    echo ERROR: run.py not found!
    echo Please ensure this script is in the GUI directory
    pause
    exit /b 1
)

if not exist "gradio_app.py" (
    echo ERROR: gradio_app.py not found!
    echo Please ensure project files are complete
    pause
    exit /b 1
)

echo Project directory check passed
echo.

:: ============================================
:: Step 3: Check and Install Dependencies
:: ============================================
echo [3/5] Checking dependencies...

if exist "requirements.txt" (
    echo Found requirements.txt, checking dependency status...
    
    :: Check if key dependencies are installed
    set "MISSING_DEPS="
    
    python -c "import gradio" >nul 2>&1
    if errorlevel 1 set "MISSING_DEPS=!MISSING_DEPS! gradio"
    
    python -c "import ultralytics" >nul 2>&1
    if errorlevel 1 set "MISSING_DEPS=!MISSING_DEPS! ultralytics"
    
    python -c "import cv2" >nul 2>&1
    if errorlevel 1 set "MISSING_DEPS=!MISSING_DEPS! opencv-python"
    
    python -c "import numpy" >nul 2>&1
    if errorlevel 1 set "MISSING_DEPS=!MISSING_DEPS! numpy"
    
    python -c "from PIL import Image" >nul 2>&1
    if errorlevel 1 set "MISSING_DEPS=!MISSING_DEPS! pillow"
    
    if defined MISSING_DEPS (
        if not "!MISSING_DEPS!"=="" (
            echo Missing dependencies detected:
            for %%p in (!MISSING_DEPS!) do echo    - %%p
            echo.
            echo Installing dependencies, please wait...
            echo.
            
            python -m pip install -r requirements.txt --upgrade
            
            if errorlevel 1 (
                echo Dependency installation failed!
                echo Please try manually: pip install -r requirements.txt
                pause
                exit /b 1
            )
            
            echo Dependencies installed successfully!
        ) else (
            echo All dependencies are installed
        )
    ) else (
        echo All dependencies are installed
    )
) else (
    echo requirements.txt not found, skipping dependency check
)
echo.

:: ============================================
:: Step 4: Create Model Directories
:: ============================================
echo [4/5] Checking model directories...

set "MODEL_DIRS=models pt_models weights yolo_models"

for %%d in (%MODEL_DIRS%) do (
    if not exist "%%d" (
        mkdir "%%d" >nul 2>&1
        echo Created directory: %%d
    )
)

:: Check parent directory for YOLO/Models
cd ..
if exist "YOLO\Models" (
    echo Found model directory: %cd%\YOLO\Models
) else (
    echo No pre-configured models found in YOLO/Models
)
cd /d "%~dp0"

echo Model directory check complete
echo.

:: ============================================
:: Step 5: Start Application
:: ============================================
echo [5/5] Starting Gradio application...
echo.
echo ============================================================
echo.
echo Service Startup Information
echo.
echo Local Access:  http://localhost:7860
echo Network Access: http://0.0.0.0:7860
echo Project Directory: %SCRIPT_DIR%
echo.
echo Note: Press Ctrl+C to stop the service
echo First startup may take time to load models...
echo.
echo ============================================================
echo.

:: Set environment variables for Python performance
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Start application
echo Starting service...
echo ============================================
echo.

python run.py

:: Capture exit code
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ============================================

if %EXIT_CODE% equ 0 (
    echo Service stopped normally
) else (
    echo Service exited with error (code: %EXIT_CODE%)
    echo.
    echo Troubleshooting:
    echo    1. Check if port 7860 is in use
    echo    2. Check if model files exist and are complete
    echo    3. Review error messages above
    echo    4. Try restarting your computer
)

echo.
echo Press any key to exit...
pause >nul

endlocal
exit /b %EXIT_CODE%
