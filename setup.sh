#!/bin/bash

set -e

ROLE=""
MASTER_IP=""

print_help() {
    echo "Usage: ./setup.sh --role master|worker [--master-ip <IP>]"
    echo
    echo "Options:"
    echo "  --role master|worker    Set node role"
    echo "  --master-ip <IP>        Master node IP address (required for worker)"
    echo "  -h, --help              Show help"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --role) ROLE="$2"; shift ;;
        --master-ip) MASTER_IP="$2"; shift ;;
        -h|--help) print_help ;;
        *) echo "Unknown parameter passed: $1"; print_help ;;
    esac
    shift
done

if [[ -z "$ROLE" ]]; then
    echo "Error: --role is required"
    exit 1
fi

if [[ "$ROLE" == "worker" && -z "$MASTER_IP" ]]; then
    echo "Error: --master-ip is required for worker"
    exit 1
fi

cd /

echo "[*] Installing Verilator..."
rm -rf verilator
git clone https://github.com/verilator/verilator
cd verilator && git checkout v5.006
autoconf && ./configure
make -j $(nproc)
make install
cd /

echo "[*] Setting up cascade-meta environment..."
source /cascade-meta/env.sh
cd /cascade-meta/design-processing
python3 -u make_all_designs.py

echo "[*] Updating objcopy configuration..."
sed -i 's/objcopy=""/objcopy="riscv64-unknown-elf-objcopy"/' /usr/local/bin/riscv64-unknown-elf-elf2hex

echo "[*] Installing Python dependencies..."
pip install "ray[default]"

if [[ "$ROLE" == "master" ]]; then
    echo "[*] Starting Ray head node..."
    ray start --head --include-dashboard=True --dashboard-host=0.0.0.0 --dashboard-port=8265
else
    echo "[*] Connecting to Ray cluster at $MASTER_IP..."
    ray start --address="$MASTER_IP:6379"
fi

echo "Setup completed for role: $ROLE"
