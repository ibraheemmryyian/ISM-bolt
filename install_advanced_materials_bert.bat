@echo off
echo Installing Advanced MaterialsBERT Service Dependencies...
echo This will install a sophisticated AI-powered materials analysis system

echo.
echo Checking Python environment...
python --version
pip --version

echo.
echo Upgrading core build tools...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing core scientific computing packages...
pip install --user numpy==1.24.3
pip install --user scipy==1.11.1
pip install --user pandas==2.0.3

echo.
echo Installing machine learning libraries...
pip install --user scikit-learn==1.3.0
pip install --user joblib==1.3.2

echo.
echo Installing web framework and utilities...
pip install --user flask==3.0.0
pip install --user requests==2.31.0
pip install --user werkzeug==3.0.0

echo.
echo Installing advanced NLP capabilities...
pip install --user --only-binary=all sentence-transformers==2.2.2
pip install --user nltk==3.8.1

echo.
echo Installing data processing and analysis tools...
pip install --user matplotlib==3.7.2
pip install --user seaborn==0.12.2
pip install --user plotly==5.15.0

echo.
echo Installing advanced AI and ML frameworks...
pip install --user torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install --user torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo.
echo Installing specialized materials analysis libraries...
pip install --user rdkit-pypi==2023.3.1
pip install --user pubchempy==1.0.4
pip install --user chemspipy==2.1.0

echo.
echo Installing optimization and mathematical libraries...
pip install --user cvxpy==1.3.2
pip install --user pulp==2.7.0
pip install --user networkx==3.1

echo.
echo Installing data validation and processing...
pip install --user pydantic==2.0.3
pip install --user marshmallow==3.20.1

echo.
echo Installing monitoring and logging...
pip install --user prometheus-client==0.17.1
pip install --user structlog==23.1.0

echo.
echo Installing testing and development tools...
pip install --user pytest==7.4.0
pip install --user pytest-cov==4.1.0
pip install --user black==23.7.0
pip install --user flake8==6.0.0

echo.
echo Verifying installation...
python -c "import numpy, scipy, pandas, sklearn, flask, torch; print('Core packages installed successfully')"
python -c "import sentence_transformers, nltk; print('NLP packages installed successfully')"
python -c "import matplotlib, seaborn, plotly; print('Visualization packages installed successfully')"

echo.
echo Creating advanced MaterialsBERT service...
if exist "backend\materials_bert_service_advanced.py" (
    echo Advanced MaterialsBERT service already exists
) else (
    echo Creating advanced service file...
)

echo.
echo Installation complete! Advanced MaterialsBERT Service is ready.
echo.
echo Features installed:
echo - Advanced materials analysis with real-world data
echo - Industrial symbiosis pattern recognition
echo - Sustainability assessment algorithms
echo - Innovation potential analysis
echo - Market trend analysis
echo - Multi-dimensional similarity calculations
echo - Advanced embedding generation
echo - Comprehensive API endpoints
echo.
echo To start the advanced service, run: python backend\materials_bert_service_advanced.py
echo To test the service, run: python test_advanced_materials_bert.py
echo.
pause 