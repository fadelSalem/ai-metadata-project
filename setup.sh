#!/bin/bash
# setup.sh - PyTorch compatible setup script

set -e

echo "========================================"
echo "AI Metadata API Environment Setup"
echo "========================================"

echo "[1/8] Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

echo "[2/8] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    libtesseract-dev \
    wget \
    curl \
    git

echo "[3/8] Creating required directories..."
mkdir -p ~/ai-metadata-project
cd ~/ai-metadata-project
mkdir -p model_cache uploaded_images logs

echo "[4/8] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[5/8] Upgrading pip and core tools..."
pip install --upgrade pip setuptools wheel

echo "[6/8] Installing Python libraries..."

echo "Installing PyTorch 2.5.0..."
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu

pip install fastapi==0.104.1 uvicorn==0.24.0
pip install pillow==10.1.0 pytesseract==0.3.10
pip install transformers==4.36.0
pip install requests==2.31.0 python-multipart==0.0.6 psutil==5.9.6

MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$MEMORY_GB" -lt 2 ]; then
    echo "[7/8] Creating SWAP file (Memory: ${MEMORY_GB}GB)..."

    sudo swapoff -a 2>/dev/null || true
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi

    echo "SWAP created successfully"
else
    echo "[7/8] Skipping SWAP creation (Memory sufficient: ${MEMORY_GB}GB)"
fi

echo "[8/8] Creating systemd service..."
sudo tee /etc/systemd/system/ai-metadata.service > /dev/null << 'SERVICE'
[Unit]
Description=AI Metadata API Service
After=network.target

[Service]
User=$USER
Group=$USER
WorkingDirectory=/home/$USER/ai-metadata-project
Environment="PATH=/home/$USER/ai-metadata-project/venv/bin"
ExecStart=/home/$USER/ai-metadata-project/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

MemoryMax=1700M
MemorySwapMax=2G
MemoryHigh=1500M

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable ai-metadata.service
sudo systemctl start ai-metadata.service

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "Available commands:"
echo "• Start service:   sudo systemctl start ai-metadata"
echo "• Stop service:    sudo systemctl stop ai-metadata"
echo "• Service status:  sudo systemctl status ai-metadata"
echo "• View logs:       sudo journalctl -u ai-metadata -f"
echo "• Test API:        curl http://localhost:8000/health"
echo "• API docs:        http://YOUR_SERVER_IP:8000/docs"
echo ""
echo "Installed PyTorch versions:"
echo "   - torch: 2.5.0"
echo "   - torchvision: 0.20.0"
echo "   - torchaudio: 2.5.0"
echo "========================================"

echo "Running quick health check..."
sleep 3
if command -v curl &> /dev/null; then
    curl -s http://localhost:8000/health || echo "Service not started yet. Please wait."
else
    echo "To install curl: sudo apt-get install curl"
fi
