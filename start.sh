#!/bin/bash
# autoRecolor startup script
# Fixes the Ollama service file, starts Ollama + autoRecolor server

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${CYAN}[autoRecolor]${NC} $1"; }
success() { echo -e "${GREEN}[autoRecolor]${NC} $1"; }
warn()    { echo -e "${YELLOW}[autoRecolor]${NC} $1"; }
error()   { echo -e "${RED}[autoRecolor]${NC} $1"; }

# ── 1. Fix & start Ollama service ─────────────────────────────────────────────
info "Writing clean Ollama service file..."
sudo tee /etc/systemd/system/ollama.service > /dev/null << 'EOF'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="NVIDIA_VISIBLE_DEVICES=all"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/lib/x86_64-linux-gnu"

[Install]
WantedBy=default.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ollama --quiet

info "Starting Ollama service..."
sudo systemctl restart ollama
sleep 2

# ── 2. Verify Ollama is up ────────────────────────────────────────────────────
for i in {1..10}; do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        success "Ollama is running"
        break
    fi
    if [ $i -eq 10 ]; then
        error "Ollama failed to start. Check: sudo systemctl status ollama"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# ── 3. Check GPU ──────────────────────────────────────────────────────────────
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$FREE_VRAM" ]; then
    success "GPU detected — ${FREE_VRAM} MiB VRAM free"
else
    warn "nvidia-smi not found — GPU status unknown"
fi

# ── 4. Pre-warm the model ─────────────────────────────────────────────────────
MODEL="qwen3.6:27b"
info "Pre-warming model ${MODEL} into VRAM (this may take ~30s on first run)..."
curl -sf http://localhost:11434/api/generate \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"keep_alive\":\"60m\"}" \
    --max-time 120 > /dev/null 2>&1 &
WARM_PID=$!

# ── 5. Start autoRecolor server ───────────────────────────────────────────────
info "Starting autoRecolor server at http://localhost:8010 ..."
autorecolor-server &
SERVER_PID=$!

# Wait for server to be ready
sleep 2
for i in {1..10}; do
    if curl -sf http://localhost:8010 > /dev/null 2>&1; then
        success "autoRecolor is ready → http://localhost:8010"
        break
    fi
    if [ $i -eq 10 ]; then
        warn "Server took longer than expected to start"
    fi
    sleep 1
done

echo ""
success "All services running. Press Ctrl+C to stop."
echo -e "  ${CYAN}Web UI${NC}  → http://localhost:8010"
echo -e "  ${CYAN}Ollama${NC}  → http://localhost:11434"
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────────
trap 'info "Shutting down..."; kill $SERVER_PID $WARM_PID 2>/dev/null; exit 0' INT TERM

wait $SERVER_PID
