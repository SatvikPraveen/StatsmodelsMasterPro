#!/bin/bash
set -e

if [ "$APP_MODE" = "jupyter" ]; then
    echo "🔵 Launching JupyterLab..."
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
else
    echo "🟢 Launching Streamlit..."
    streamlit run Home.py
fi
