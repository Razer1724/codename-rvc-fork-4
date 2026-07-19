@echo off
setlocal

:: --- CHECKS ---
:: 1. Did the user drop files?
if "%~1" == "" goto :ErrorNoFiles

:: 2. Check for 'python' command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=python
    goto :FoundPython
)

:: 3. Check for 'py' launcher
py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=py
    goto :FoundPython
)

:: If we get here, no Python was found
goto :ErrorNoPython

:FoundPython
:: --- EXECUTION ---
echo ---------------------------------------------------
echo Python found: %PYCMD%
echo Processing dropped files...
echo ---------------------------------------------------

:: Pass ALL dropped files (%*) to the script
"%PYCMD%" "%~dp0concat.py" %*

if %errorlevel% neq 0 (
    echo.
    echo [!] The Python script encountered an error.
)

echo.
echo ---------------------------------------------------
echo Done.
pause
exit /b

:: --- ERROR HANDLERS ---
:ErrorNoFiles
echo [!] NO FILES DETECTED.
echo Please drag and drop a folder OR a selection of WAV files.
echo.
pause
exit /b

:ErrorNoPython
echo [X] CRITICAL ERROR: Python not found.
echo Please install it from python.org and check "Add to PATH".
pause
exit /b