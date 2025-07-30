@echo off
echo.
echo ========================================
echo   SYMBIOFLOWS PERSISTENT DEMO SYSTEM
echo ========================================
echo.

echo Starting SymbioFlows Persistent Demo...
echo This window will remain open to keep services running.
echo Press Ctrl+C to stop all services when done.
echo.

python keep_running_demo.py

echo.
echo Services have been terminated.
echo.
pause