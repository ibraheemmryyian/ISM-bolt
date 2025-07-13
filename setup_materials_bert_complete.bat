@echo off
echo ========================================
echo COMPLETE MATERIALSBERT SETUP
echo ========================================

echo.
echo This script will:
echo 1. Install all required dependencies
echo 2. Start the MaterialsBERT service
echo 3. Test the service
echo.

echo Starting installation...
call install_materials_bert.bat

echo.
echo Installation complete! Starting service...
echo.

python start_materials_bert_simple.py

echo.
echo Setup complete!
echo.

pause 