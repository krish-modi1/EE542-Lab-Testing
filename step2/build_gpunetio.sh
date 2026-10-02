#!/usr/bin/env bash
# Step 3: build NVIDIA's open-source DOCA GPUNetIO and run one of its examples.
# Expected result: the build succeeds and the example fails, because this machine has no
# RDMA network card (ConnectX-6 Dx or newer, or BlueField). The logs are the evidence.
#
# Usage: bash build_gpunetio.sh [CUDA_ARCH]
#   86 = RTX 30xx / A10 / A30, 89 = RTX 40xx / L4, 90 = H100
ARCH=${1:-86}
mkdir -p logs
sudo apt-get install -y rdma-core libibverbs-dev ibverbs-utils

# Which RDMA devices and network cards does this machine have?
{ echo "RDMA devices:"; ibv_devices; echo "Network cards:"; lspci | grep -i ethernet; } \
  2>&1 | tee logs/rdma_devices.log

# Build the library and its examples for this GPU. The GPUNetIO Makefile assumes
# CUDA_HOME=/usr/local/cuda; Ubuntu's nvidia-cuda-toolkit puts nvcc under /usr instead.
CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
[ -d gpunetio ] || git clone --depth 1 https://github.com/NVIDIA-DOCA/gpunetio.git
make -C gpunetio -j"$(nproc)" install install_examples PREFIX="$PWD/gpunetio/install" \
  CUDA_ARCH="$ARCH" CUDA_HOME="$CUDA_HOME" 2>&1 | tee logs/gpunetio_build.log

# Run one example. nvidia-smi prints the GPU as 00000000:07:00.0; the example wants 07:00.0.
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1 | cut -d: -f2- | tr 'A-F' 'a-f')
LD_LIBRARY_PATH="$PWD/gpunetio/install/lib" DOCA_GPUNETIO_LOG=6 \
  gpunetio/install/examples/gpunetio_verbs_write_lat -d mlx5_0 -g "$GPU_PCI" \
  2>&1 | tee logs/gpunetio_run.log
