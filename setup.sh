#!/usr/bin/env bash

set -euo pipefail

info() {
  echo -e "\033[1;34m[INFO]\033[0m $*"
}

warn() {
  echo -e "\033[1;33m[WARN]\033[0m $*"
}

error() {
  echo -e "\033[1;31m[ERROR]\033[0m $*"
}

# 1. Check for Python installation
info "Checking for Python installation..."
PYTHON_CMD=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  error "Python is not installed. Please download and install Python 3.11+ from https://www.python.org/downloads/"
  exit 1
fi

info "Found Python executable: ${PYTHON_CMD}"

# 2. Check for uv installation
info "Checking for uv installation..."
if ! command -v uv >/dev/null 2>&1; then
  error "uv is not installed. Please follow the installation guide at https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

info "uv is installed."

# 3. Create virtual environment using uv
info "Creating virtual environment in .venv (if missing)..."
uv venv

# 4. Activate the virtual environment for the remainder of the script
if [[ -f .venv/bin/activate ]]; then
  info "Activating virtual environment..."
  # shellcheck source=/dev/null
  source .venv/bin/activate
else
  error "Virtual environment activation script not found at .venv/bin/activate"
  exit 1
fi

# 5. Install Python dependencies via uv pip
if [[ -f requirements.txt ]]; then
  info "Installing dependencies from requirements.txt using uv pip..."
  uv pip install -r requirements.txt
else
  warn "requirements.txt not found. Skipping dependency installation."
fi

# 6 & 7. Ensure Models and filled_models directories exist
info "Ensuring Models directory exists..."
mkdir -p Models

info "Ensuring filled_models directory exists..."
mkdir -p filled_models

info "Setup complete. You are now ready to use LLM Score Engine."
echo "To activate the virtual environment in a new shell, run: source .venv/bin/activate"
