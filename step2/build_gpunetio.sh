#!/usr/bin/env bash
# Step 2d: build NVIDIA's open-source DOCA GPUNetIO and try to run one example.
# Expected result: the build succeeds, and the example fails because this machine has no
# RDMA-capable NVIDIA NIC (ConnectX-6 Dx or newer, or BlueField). The logs are evidence
# for the hardware blocker.
#
# Usage: bash build_gpunetio.sh [CUDA_ARCH]   (default 86 = Ampere, e.g. RTX 3060)
set -u
ARCH=${1:-86}
LOG=$(pwd)/logs
mkdir -p "$LOG"
export PATH=/usr/local/cuda/bin:$PATH

echo "== installing rdma-core / libibverbs =="
apt-get install -y rdma-core libibverbs-dev ibverbs-utils >/dev/null 2>&1 || \
  sudo apt-get install -y rdma-core libibverbs-dev ibverbs-utils

echo "== RDMA devices on this machine ==" | tee "$LOG/rdma_devices.log"
ibv_devices 2>&1 | tee -a "$LOG/rdma_devices.log"
ls /sys/class/infiniband 2>&1 | tee -a "$LOG/rdma_devices.log"
lspci 2>/dev/null | grep -iE "mellanox|nvidia.*(connectx|bluefield)|ethernet" | tee -a "$LOG/rdma_devices.log"

echo "== cloning and building gpunetio (CUDA_ARCH=$ARCH) =="
[ -d gpunetio ] || git clone --depth 1 https://github.com/NVIDIA-DOCA/gpunetio.git
cd gpunetio
make -j"$(nproc)" install install_examples PREFIX="$(pwd)/install" CUDA_ARCH="$ARCH" \
  2>&1 | tee "$LOG/gpunetio_build.log"
echo "build exit code: ${PIPESTATUS[0]}" | tee -a "$LOG/gpunetio_build.log"

echo "== running an example (expected to fail without a ConnectX/BlueField NIC) =="
EX=$(find install . -type f -name 'gpunetio_verbs_write_lat' -perm -u+x 2>/dev/null | head -n 1)
# nvidia-smi prints e.g. 00000000:8A:00.0; the examples expect 8a:00.0
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1 | sed -E 's/^[0-9A-Fa-f]+://' | tr 'A-F' 'a-f')
if [ -z "$EX" ]; then
  echo "example binary not found (build may have failed); see logs/gpunetio_build.log" | tee "$LOG/gpunetio_run.log"
else
  echo "running: $EX -d mlx5_0 -g $GPU_PCI" | tee "$LOG/gpunetio_run.log"
  DOCA_GPUNETIO_LOG=6 timeout 30 "$EX" -d mlx5_0 -g "$GPU_PCI" 2>&1 | tee -a "$LOG/gpunetio_run.log"
  echo "exit code: ${PIPESTATUS[0]}" | tee -a "$LOG/gpunetio_run.log"
fi
echo "logs written to $LOG/"
