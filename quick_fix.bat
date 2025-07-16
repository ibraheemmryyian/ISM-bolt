@echo off
echo ========================================
echo Quick Fix for Import Issues
echo ========================================
echo.

echo Fixing sentence-transformers import issue...
pip uninstall -y sentence-transformers transformers huggingface-hub
pip install --user huggingface-hub==0.20.3
pip install --user transformers==4.35.0
pip install --user sentence-transformers==2.2.2

echo.
echo Fixing sklearn import issue...
pip uninstall -y scikit-learn numpy
pip install --user numpy==1.24.3
pip install --user scikit-learn==1.3.0

echo.
echo Testing imports...
python -c "import sentence_transformers; print('✅ sentence-transformers works')"
python -c "import sklearn; print('✅ sklearn works')"

echo.
echo Quick fix completed!
pause 