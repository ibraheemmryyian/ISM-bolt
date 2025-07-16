@echo off
echo ========================================
echo ISM AI System Production Test Suite
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Running comprehensive production tests...
echo This will test all components of the AI system
echo.

python production_test_suite.py

echo.
echo Production tests completed!
echo Check production_test_report.json for detailed results
echo.
pause 