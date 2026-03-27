@echo off
REM run.bat -- Quick launcher for Windows
REM Usage: double-click or run from command prompt

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Retail Shelf Intelligence System...
python src/main.py %*
pause
