@echo off
echo ========================================
echo Fix NumPy Version Conflicts
echo ========================================
echo.

echo Uninstalling conflicting packages...
pip uninstall -y numpy scipy scikit-learn sentence-transformers transformers huggingface-hub thinc

echo.
echo Installing compatible versions in order...

echo 1. Installing NumPy 1.24.3...
pip install --user --no-deps numpy==1.24.3

echo 2. Installing SciPy 1.11.1...
pip install --user --no-deps scipy==1.11.1

echo 3. Installing scikit-learn 1.3.0...
pip install --user --no-deps scikit-learn==1.3.0

echo 4. Installing huggingface-hub 0.20.3...
pip install --user --no-deps huggingface-hub==0.20.3

echo 5. Installing transformers 4.35.0...
pip install --user --no-deps transformers==4.35.0

echo 6. Installing sentence-transformers 2.2.2...
pip install --user --no-deps sentence-transformers==2.2.2

echo.
echo Testing imports...
python -c "import numpy; print('✅ NumPy version:', numpy.__version__)"
python -c "import sklearn; print('✅ sklearn works')"
python -c "import sentence_transformers; print('✅ sentence-transformers works')"

echo.
echo NumPy conflicts fixed!
pause 