@echo off
echo 🚀 Starting Agentic 3D Financial Intelligence Dashboard...
echo.
start cmd /k "uvicorn api:app --reload --port 8000"
timeout /t 2
start cmd /k "streamlit run unified_app.py --server.port 8502"
echo.
echo ✅ Services starting!
echo FastAPI Docs  → http://localhost:8000/docs
echo Unified 3D    → http://localhost:8502
echo.
pause
