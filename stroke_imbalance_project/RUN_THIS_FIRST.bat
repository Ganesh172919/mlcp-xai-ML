@echo off
echo ===============================================================
echo STROKE PREDICTION PROJECT - COMPLETE SETUP
echo ===============================================================
echo.
echo This will create the complete project structure with all files.
echo.
pause

echo.
echo Step 1: Running BUILD_COMPLETE_PROJECT.py...
python BUILD_COMPLETE_PROJECT.py

echo.
echo Step 2: Running generate_complete_project.py...
python generate_complete_project.py

echo.
echo ===============================================================
echo SETUP COMPLETE!
echo ===============================================================
echo.
echo Next steps:
echo 1. Download stroke dataset from Kaggle
echo 2. Place as data\stroke.csv
echo 3. Install requirements: pip install -r requirements.txt
echo 4. Run notebooks in order
echo.
pause
