@echo off
echo ================================================================================
echo 🔧 COMPREHENSIVE MATCH QUALITY FIX AND IMPROVEMENT
echo ================================================================================
echo.
echo This script will:
echo 1. Fix existing problematic matches (duplicates, generic names, etc.)
echo 2. Generate new high-quality matches using improved AI engine
echo 3. Validate results and provide comprehensive reporting
echo.
echo Starting process...
echo.

cd /d "%~dp0backend"

python fix_and_improve_matches.py

echo.
echo ================================================================================
echo Process completed! Check the output above for results.
echo ================================================================================
pause 