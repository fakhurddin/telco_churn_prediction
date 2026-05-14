@echo off
echo ============================================================
echo   Customer Churn Predictor - Full Pipeline
echo ============================================================

echo.
echo [1/5] Running Data Cleaning...
python src/data_cleaning.py
if %errorlevel% neq 0 (
    echo ERROR: data_cleaning.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [2/5] Running EDA...
python src/eda.py
if %errorlevel% neq 0 (
    echo ERROR: eda.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [3/5] Running Feature Engineering...
python src/feature_engineering.py
if %errorlevel% neq 0 (
    echo ERROR: feature_engineering.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo [4/5] Training Model...
python src/train_model.py
if %errorlevel% neq 0 (
    echo ERROR: train_model.py failed. Stopping.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   All phases complete! Launching Streamlit app...
echo ============================================================
echo.
echo [5/5] Starting Streamlit...
echo   App will open at: http://localhost:8501
echo   Press Ctrl+C to stop the app
echo.
streamlit run app.py
