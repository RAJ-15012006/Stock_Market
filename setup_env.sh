#!/bin/bash
# ============================================================
#  Stock Market AI Platform — Environment Setup Script
#  Run this ONCE to create a fresh virtual environment.
#  Usage:  bash setup_env.sh
# ============================================================

set -e

PYTHON="/opt/homebrew/bin/python3.11"
VENV_DIR="venv"

echo ""
echo "=========================================="
echo "  Stock Market AI Platform — Setup"
echo "=========================================="
echo ""

# 1. Create virtual environment
echo "[1/4] Creating virtual environment at './$VENV_DIR' ..."
$PYTHON -m venv $VENV_DIR
echo "      Done."

# 2. Upgrade pip inside venv
echo "[2/4] Upgrading pip ..."
$VENV_DIR/bin/pip install --upgrade pip --quiet
echo "      Done."

# 3. Install all project requirements
echo "[3/4] Installing requirements (this may take a few minutes) ..."
$VENV_DIR/bin/pip install -r requirements.txt --quiet
echo "      Done."

# 4. Quick import check
echo "[4/4] Verifying key imports ..."
$VENV_DIR/bin/python -c "import agno, groq, streamlit, yfinance, plotly, numpy, sklearn, shap; print('      All imports OK.')"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "  Run the app with:  bash run.sh"
echo "=========================================="
echo ""
