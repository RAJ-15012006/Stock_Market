#!/bin/bash
# ============================================================
#  Stock Market AI Platform — Run Script
#  Usage:  bash run.sh
# ============================================================

VENV_DIR="venv"

# If venv doesn't exist, run setup first
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment not found. Running setup first..."
    bash setup_env.sh
fi

echo ""
echo "=========================================="
echo "  Starting Stock Market AI Platform..."
echo "  Open: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo "=========================================="
echo ""

# Activate and run
source $VENV_DIR/bin/activate
python -m streamlit run unified_app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false
