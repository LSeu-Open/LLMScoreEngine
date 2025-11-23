@echo off
setlocal enabledelayedexpansion

call :info Checking for Python installation...
where python >nul 2>&1
if errorlevel 1 (
  call :error Python is not installed. Please download and install Python 3.11+ from https://www.python.org/downloads/
  exit /b 1
)
for /f "delims=" %%i in ('where python 2^>nul') do (
  set "PYTHON_CMD=%%i"
  goto after_python_detect
)
:after_python_detect
if not defined PYTHON_CMD set PYTHON_CMD=python
call :info Found Python executable: %PYTHON_CMD%

call :info Checking for uv installation...
where uv >nul 2>&1
if errorlevel 1 (
  call :error uv is not installed. Please follow the guide at https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)
call :info uv is installed.

call :info Creating virtual environment in .venv (if missing)...
uv venv

if exist .venv\Scripts\activate.bat (
  call :info Activating virtual environment...
  call .venv\Scripts\activate.bat
) else (
  call :error Virtual environment activation script not found at .venv\Scripts\activate.bat
  exit /b 1
)

if exist requirements.txt (
  call :info Installing dependencies from requirements.txt using uv pip...
  uv pip install -r requirements.txt
) else (
  call :warn requirements.txt not found. Skipping dependency installation.
)

if not exist Models (
  call :info Creating Models directory...
  mkdir Models >nul 2>&1
)

if not exist filled_models (
  call :info Creating filled_models directory...
  mkdir filled_models >nul 2>&1
)

call :info Setup complete. You are now ready to use LLM Score Engine.
echo To activate the virtual environment in a new shell, run: call .venv\Scripts\activate.bat
exit /b 0

:info
  echo [INFO] %*
  exit /b 0

:warn
  echo [WARN] %*
  exit /b 0

:error
  echo [ERROR] %*
  exit /b 0
