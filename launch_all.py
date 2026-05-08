import subprocess
import sys
import os
import time
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

processes = []

def start(cmd, cwd=BASE):
    p = subprocess.Popen(cmd, cwd=cwd, shell=True)
    processes.append(p)
    return p

print("🚀 Starting all services...\n")

start(f'uvicorn api:app --reload --port 8000')
time.sleep(2)

start(f'streamlit run app.py --server.port 8501')
time.sleep(2)

start(f'{PYTHON} playground.py')
time.sleep(2)

start(f'streamlit run unified_app.py --server.port 8502')
time.sleep(4)

print("\n✅ All services running:")
print("  FastAPI Docs  → http://localhost:8000/docs")
print("  Streamlit App → http://localhost:8501")
print("  Playground    → http://localhost:7777")
print("  Unified 3D    → http://localhost:8502")

webbrowser.open("http://localhost:8000/docs")
webbrowser.open("http://localhost:8501")
webbrowser.open("http://localhost:8502")

print("\nPress Ctrl+C to stop all services.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Shutting down all services...")
    for p in processes:
        p.terminate()
