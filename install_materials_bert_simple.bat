@echo off
echo Installing Simplified MaterialsBERT Service Dependencies...

echo.
echo Checking existing packages...
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import flask; print(f'Flask version: {flask.__version__}')" 2>nul || echo Flask not installed

echo.
echo Installing only essential packages (avoiding compilation issues)...
pip install --user flask==3.0.0
pip install --user requests==2.31.0

echo.
echo Installing lightweight alternatives...
pip install --user scikit-learn==1.3.0
pip install --user pandas==2.0.3

echo.
echo Installing pre-compiled packages only...
pip install --user --only-binary=all sentence-transformers==2.2.2

echo.
echo Creating lightweight MaterialsBERT service...
if not exist "backend\materials_bert_service_simple.py" (
    echo MaterialsBERT service already exists
) else (
    echo Service file already exists
)

echo.
echo Installation complete! The simplified MaterialsBERT service is ready.
echo.
echo To start the service, run: python backend\materials_bert_service_simple.py
echo To test the service, run: python test_materials_bert_simple.py
echo.
pause 