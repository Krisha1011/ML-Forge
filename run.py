import subprocess
import sys
import os
import time

def main():
    print("==================================================")
    print("          MLForge: AutoML Web Platform            ")
    print("==================================================")
    
    # Resolve venv python interpreter path
    venv_python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        # Fallback to general python if venv isn't set up yet
        venv_python = "python"
        print("Warning: Virtual environment python not found at .venv/Scripts/python.exe. Using system python.")

    # 1. Spin up FastAPI backend
    print("Starting FastAPI Backend on port 8000...")
    backend_cmd = [
        venv_python, "-m", "uvicorn", "backend.app.main:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ]
    
    backend_proc = subprocess.Popen(
        backend_cmd,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    
    # Wait for backend to launch
    time.sleep(2)
    
    # 2. Spin up Vite frontend
    print("Starting Vite Frontend on port 3000...")
    # On Windows, we run npm through shell=True to load its cmd wrapper
    frontend_proc = subprocess.Popen(
        "npm run dev",
        cwd="frontend",
        shell=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    
    print("\n[MLForge Server Control Panel]")
    print("==================================================")
    print("-> React App Interface: http://localhost:3000")
    print("-> FastAPI Documentation: http://localhost:8000/docs")
    print("==================================================")
    print("Press Ctrl+C in this terminal to safely terminate both servers.\n")
    
    # Read process streams in non-blocking way or print lines
    try:
        while True:
            # We poll backend and frontend streams
            # If backend or frontend exits, terminate the other
            if backend_proc.poll() is not None:
                print(f"Backend exited with code {backend_proc.poll()}")
                break
            if frontend_proc.poll() is not None:
                print(f"Frontend exited with code {frontend_proc.poll()}")
                break
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nShutting down MLForge server instances...")
    finally:
        # Gracefully kill processes
        backend_proc.terminate()
        frontend_proc.terminate()
        try:
            backend_proc.wait(timeout=3)
            frontend_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
            frontend_proc.kill()
        print("MLForge servers shut down successfully.")

if __name__ == "__main__":
    main()
