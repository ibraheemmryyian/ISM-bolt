@echo off
echo ========================================
echo   ISM AI - COMPLETE SYSTEM SETUP
echo ========================================
echo.

echo [1/5] Importing 50 Gulf Companies...
python backend/real_data_bulk_importer.py
if %errorlevel% neq 0 (
    echo ❌ Failed to import companies
    pause
    exit /b 1
)
echo ✅ Companies imported successfully
echo.

echo [2/5] Generating AI Listings for all companies...
curl -X POST http://localhost:5000/api/ai/generate-all-listings
if %errorlevel% neq 0 (
    echo ❌ Failed to generate AI listings
    pause
    exit /b 1
)
echo ✅ AI listings generated successfully
echo.

echo [3/5] Running Advanced AI Matching Engine...
python backend/revolutionary_ai_matching.py
if %errorlevel% neq 0 (
    echo ❌ Failed to run AI matching
    pause
    exit /b 1
)
echo ✅ AI matching completed successfully
echo.

echo [4/5] Running GNN Reasoning Engine...
python backend/gnn_reasoning_engine.py
if %errorlevel% neq 0 (
    echo ❌ Failed to run GNN reasoning
    pause
    exit /b 1
)
echo ✅ GNN reasoning completed successfully
echo.

echo [5/5] Running Multi-Hop Symbiosis Analysis...
python backend/multi_hop_symbiosis_service.py
if %errorlevel% neq 0 (
    echo ❌ Failed to run symbiosis analysis
    pause
    exit /b 1
)
echo ✅ Symbiosis analysis completed successfully
echo.

echo ========================================
echo   🎉 SYSTEM SETUP COMPLETE!
echo ========================================
echo.
echo 📊 What was generated:
echo    - 50 Gulf companies imported
echo    - AI-generated material listings
echo    - Advanced AI matches using 4-factor matching
echo    - GNN-based reasoning and insights
echo    - Multi-hop symbiosis networks
echo.
echo 🔗 Access your system:
echo    - Frontend: http://localhost:3000
echo    - Backend API: http://localhost:5000
echo    - Admin Dashboard: Check the frontend
echo.
echo 🧪 Test the system:
echo    - Browse companies and their AI listings
echo    - View AI-generated matches
echo    - Explore symbiosis networks
echo    - Use admin features to upgrade users
echo.
pause 