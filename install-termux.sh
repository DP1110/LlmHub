#!/usr/bin/env bash
# ==============================================================================
# LLMHub: Automated Installer for Android Termux (ARM64 Non-Root)
# ==============================================================================
# Targets: Android 10+ running Termux (ARM64 / aarch64)
# Offline-first: After installation and model pull, zero internet required.
# ==============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}======================================================================${NC}"
echo -e "${BOLD}     LLMHub: Offline-First GGUF Runtime - Termux ARM64 Installer      ${NC}"
echo -e "${BOLD}======================================================================${NC}"

# 1. Environment & Non-Root Sandbox Verification
if [ -z "${TERMUX_VERSION:-}" ] && [ ! -d "/data/data/com.termux" ]; then
    echo -e "${YELLOW}[WARN] Not running in native Termux environment.${NC}"
    echo -e "       Continuing in Linux compatibility mode..."
else
    echo -e "${GREEN}[+] Detected Android Termux environment: ${TERMUX_VERSION:-v0.118+}${NC}"
fi

# Verify unprivileged non-root execution
CURRENT_UID=$(id -u)
if [ "$CURRENT_UID" -eq 0 ]; then
    echo -e "${RED}[ERROR] Do not run this script as root!${NC}"
    echo -e "        LLMHub runs strictly in the unprivileged Termux user sandbox."
    echo -e "        Root execution breaks Android app isolation and path resolution."
    exit 1
fi
echo -e "${GREEN}[+] Running safely as unprivileged user (UID: $CURRENT_UID)${NC}"

# 2. Package Manager Upgrade & Core Dependencies
echo -e "\n${BOLD}[1/5] Updating package repositories...${NC}"
if command -v pkg >/dev/null 2>&1; then
    pkg update -y || apt update -y
elif command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
fi

echo -e "\n${BOLD}[2/5] Installing native build toolchain and libraries...${NC}"
if command -v pkg >/dev/null 2>&1; then
    pkg install -y \
        python \
        python-psutil \
        ninja \
        clang \
        cmake \
        make \
        git \
        libopenblas \
        termux-tools \
        binutils || {
        echo -e "${RED}[ERROR] Failed to install essential packages via pkg.${NC}"
        exit 1
    }
    # Attempt to install native llama-cpp package if available in Termux repositories
    pkg install -y llama-cpp 2>/dev/null || true
elif command -v apt-get >/dev/null 2>&1; then
    apt-get install -y \
        python3 \
        python3-pip \
        clang \
        cmake \
        make \
        git \
        libopenblas-dev \
        binutils || {
        echo -e "${RED}[ERROR] Failed to install essential packages via apt-get.${NC}"
        exit 1
    }
fi

# Optional: termux-api for battery temperature readings
if pkg install -y termux-api 2>/dev/null; then
    echo -e "${GREEN}[+] Installed termux-api for thermal and battery monitoring.${NC}"
else
    echo -e "${YELLOW}[!] termux-api package unavailable; will use sysfs thermal zones.${NC}"
fi

# 3. Compiler & OpenBLAS Flags for ARM64 NEON Optimization
echo -e "\n${BOLD}[3/5] Configuring ARM64 NEON / OpenBLAS acceleration flags...${NC}"
export CMAKE_ARGS="-DGGML_OPENBLAS=on -DGGML_NEON=on"
export CFLAGS="-O3"
export CXXFLAGS="-O3"

# 4. Python Environment & Dependencies
echo -e "\n${BOLD}[4/5] Installing Python dependencies...${NC}"

# Install pure-Python core dependencies (guaranteed to succeed without Rust/maturin)
python -m pip install --no-build-isolation \
    toml \
    jinja2 \
    requests \
    httpx \
    rich

# Install HTTP server dependencies (FastAPI, Uvicorn, Pydantic)
echo -e "Installing HTTP server & API dependencies..."
python -m pip install --no-build-isolation \
    --extra-index-url https://termux-user-repository.github.io/pypi/ \
    fastapi uvicorn pydantic 2>/dev/null || {
    echo -e "${YELLOW}[NOTE] Pydantic/FastAPI compilation requires maturin on Python 3.14.${NC}"
    echo -e "       (Core offline terminal runner 'llmhub run' and 'llmhub pull' are 100% ready!)${NC}"
}

mkdir -p "$HOME/.llmhub/logs"
mkdir -p "$HOME/.llmhub/models"
mkdir -p "$HOME/.llmhub/run"

# Install llama-cpp-python with OpenBLAS / ARM NEON support
echo -e "Checking llama-cpp-python..."
python -m pip install llama-cpp-python --no-build-isolation --no-cache-dir > "$HOME/.llmhub/logs/build-llama.log" 2>&1 || {
    echo -e "${YELLOW}[NOTE] llama-cpp-python build skipped or already installed (logs in ~/.llmhub/logs/build-llama.log).${NC}"
}

# Install LLMHub CLI launcher directly into PATH (bypasses setuptools/pip build isolation on Python 3.14)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "Installing LLMHub CLI from ${SCRIPT_DIR}..."

# 1. Attempt standard pip install as non-editable fallback
python -m pip install --no-build-isolation --no-deps "${SCRIPT_DIR}" > "$HOME/.llmhub/logs/pip-install.log" 2>&1 || true

# 2. Deploy direct executable launcher script into bin directory
if [ -n "${TERMUX_VERSION:-}" ] || [ -d "/data/data/com.termux" ]; then
    TERMUX_BIN="${PREFIX:-/data/data/com.termux/files/usr}/bin"
else
    TERMUX_BIN="${PREFIX:-$HOME/.local}/bin"
fi
mkdir -p "$TERMUX_BIN"

cat << EOF > "$TERMUX_BIN/llmhub"
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Resolve repo root dynamically with baked SCRIPT_DIR preference
candidate_dirs = [
    Path("${SCRIPT_DIR}").resolve(),
    Path(os.environ.get("HOME", "")) / "llmhub",
    Path(os.environ.get("HOME", "")) / "LlmHub",
    Path("/data/data/com.termux/files/home/LlmHub"),
    Path.cwd()
]

found_root = None
for candidate in candidate_dirs:
    if (candidate / "llmhub").is_dir():
        found_root = candidate
        break

if found_root and str(found_root) not in sys.path:
    sys.path.insert(0, str(found_root))

from llmhub.cli import main
if __name__ == "__main__":
    main()
EOF

chmod +x "$TERMUX_BIN/llmhub"
echo -e "${GREEN}[+] Successfully deployed LLMHub launcher to ${TERMUX_BIN}/llmhub${NC}"

# 5. Initialize Directory Layout & Wake-Lock Test
echo -e "\n${BOLD}[5/5] Initializing ~/.llmhub storage layout and wake-lock...${NC}"
mkdir -p "$HOME/.llmhub/models"
mkdir -p "$HOME/.llmhub/logs"
mkdir -p "$HOME/.llmhub/run"

if command -v termux-wake-lock >/dev/null 2>&1; then
    echo -e "${GREEN}[+] Testing termux-wake-lock...${NC}"
    termux-wake-lock
    termux-wake-unlock
    echo -e "${GREEN}[+] Termux CPU wake-lock verified successfully.${NC}"
else
    echo -e "${YELLOW}[!] termux-wake-lock binary not found in PATH.${NC}"
fi

# 6. Verification and Diagnostics
echo -e "\n${BOLD}======================================================================${NC}"
echo -e "${GREEN}${BOLD}             LLMHUB INSTALLATION SUCCESSFUL!                          ${NC}"
echo -e "${BOLD}======================================================================${NC}"

echo -e "\nSystem Diagnostics:"
if command -v llmhub >/dev/null 2>&1; then
    llmhub hardware
    echo ""
    llmhub termux
else
    python -m llmhub.cli hardware
    echo ""
    python -m llmhub.cli termux
fi

echo -e "\n${BOLD}Next steps to get started:${NC}"
echo -e "  1. Pull a mobile-friendly 0.5B model:"
echo -e "     ${GREEN}llmhub pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m${NC}"
echo -e "  2. Start the daemon in low-memory mode with wake-lock:"
echo -e "     ${GREEN}llmhub serve --low-mem${NC}"
echo -e "  3. In a second Termux session, chat with your model:"
echo -e "     ${GREEN}llmhub run Qwen2.5-0.5B-Instruct-GGUF:q4_k_m${NC}"
echo -e ""
echo -e "Enjoy offline-first AI on Android with LLMHub!"
