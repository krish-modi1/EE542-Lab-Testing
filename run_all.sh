#!/usr/bin/env bash
# Run the whole DOCA lab test top to bottom and write everything to one log file.
#
#   Setup    install system packages, CUDA compiler, Docker GPU support, Python venv
#   Step 1   TensorRT ResNet-18 reading its input from GPU memory
#   Step 2   images as UDP packets -> DPDK (pcap, no NIC) -> GPU inference
#   Step 3   check DOCA hardware requirements; build NVIDIA's open-source GPUNetIO and run one example
#   Step 4   NVIDIA's official DOCA container: list devices, build and run a GPUNetIO sample
#
# Usage:  bash run_all.sh                 everything
#         SKIP_SETUP=1 bash run_all.sh    skip installs (second run onwards)
# Output: logs/run_all_<date>.log, with a summary at the end.
set -o pipefail
cd "$(dirname "$0")" || exit 1
ROOT=$PWD
VENV=${VENV:-$HOME/venv}
DOCA_IMG=${DOCA_IMG:-nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host}
mkdir -p logs step2/logs
LOG="$ROOT/logs/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

RESULTS=()
section() { echo; echo "================ $* ================"; date; }

# check NAME CMD...  run a step that should work
check() {
	local name=$1; shift
	section "$name"
	echo "+ $*"
	if "$@"; then RESULTS+=("PASS     $name"); else RESULTS+=("FAIL     $name (exit $?)"); fi
}

# blocker NAME CMD...  run a step that needs DOCA hardware; failure is the expected result
blocker() {
	local name=$1; shift
	section "$name"
	echo "+ $*"
	if "$@"; then RESULTS+=("PASS     $name (unexpected: look at this step)"); else RESULTS+=("BLOCKED  $name"); fi
}

# ---------------------------------------------------------------- machine
section "Machine"
uname -a
grep PRETTY_NAME /etc/os-release
nvidia-smi
echo "NVIDIA network cards (ConnectX / BlueField):"
lspci | grep -i -E 'mellanox|connectx|bluefield' || echo "  none"
echo "All network cards:"
lspci | grep -i -E 'ethernet|network' || echo "  none visible (normal in WSL and some VMs)"

# ---------------------------------------------------------------- setup
install_system() {
	. /etc/os-release
	local apt="sudo apt-get -o DPkg::Lock::Timeout=900"
	$apt update
	$apt install -y build-essential pkg-config git wget curl pciutils python3-venv \
		dpdk dpdk-dev libdpdk-dev libpcap-dev rdma-core libibverbs-dev ibverbs-utils || return 1
	if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
		wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb" &&
			sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb &&
			$apt update && $apt install -y cuda-toolkit-12-6 || return 1
	fi
}

install_docker_gpu() {
	local apt="sudo apt-get -o DPkg::Lock::Timeout=900"
	command -v docker >/dev/null || $apt install -y docker.io || return 1
	if ! sudo docker run --rm --gpus all ubuntu nvidia-smi -L; then
		curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
			sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
		curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
			sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' |
			sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
		$apt update && $apt install -y nvidia-container-toolkit &&
			sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
	fi
}

install_python() {
	[ -d "$VENV" ] || python3 -m venv "$VENV"
	"$VENV/bin/pip" install -q torch torchvision onnx onnxscript tensorrt scapy numpy
}

if [ -z "$SKIP_SETUP" ]; then
	check "Setup: system packages and CUDA compiler" install_system
	check "Setup: Docker with GPU access" install_docker_gpu
	check "Setup: Python packages" install_python
fi
export PATH=/usr/local/cuda/bin:$PATH
source "$VENV/bin/activate"

# ---------------------------------------------------------------- step 1
cd "$ROOT/step1" || exit 1
check "Step 1: versions" bash setup_check.sh
check "Step 1: export ResNet-18 to ONNX (fp32, fp16)" python3 export_onnx.py
check "Step 1: build TensorRT engine fp32" python3 build_engine.py fp32
check "Step 1: build TensorRT engine fp16" python3 build_engine.py fp16
check "Step 1: benchmark TensorRT vs PyTorch from GPU buffers" python3 gpu_buffer_runner.py

# ---------------------------------------------------------------- step 2
cd "$ROOT/step2" || exit 1
DPDK_ARGS=(-l 0 --no-huge -m 512 --no-pci --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null')
check "Step 2: build DPDK receiver" make
check "Step 2: make packets (in order)" python3 make_pcap.py
check "Step 2: DPDK receive and rebuild (in order)" sudo ./dpdk_rx "${DPDK_ARGS[@]}" -- frames_rx.bin
check "Step 2: classify received images (in order)" python3 infer_rx.py
check "Step 2: make packets (shuffled)" python3 make_pcap.py --shuffle
check "Step 2: DPDK receive and rebuild (shuffled)" sudo ./dpdk_rx "${DPDK_ARGS[@]}" -- frames_rx.bin
check "Step 2: classify received images (shuffled)" python3 infer_rx.py

# ---------------------------------------------------------------- step 3
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d .)
blocker "Step 3: machine meets DOCA GPUNetIO requirements" bash check_doca_requirements.sh
section "Step 3: open-source GPUNetIO (GPU arch $GPU_ARCH)"
bash build_gpunetio.sh "$GPU_ARCH"
check "Step 3: GPUNetIO builds" test -x gpunetio/install/examples/gpunetio_verbs_write_lat
blocker "Step 3: GPUNetIO example runs" bash -c '[ -f logs/gpunetio_run.log ] && ! grep -q -i -E "failed|error" logs/gpunetio_run.log'

# ---------------------------------------------------------------- step 4
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1 | cut -d: -f2- | tr 'A-F' 'a-f')
NIC_PCI=$(lspci | grep -i -E 'ethernet|network' | head -n 1 | cut -d' ' -f1)
NIC_PCI=${NIC_PCI:-00:00.0}
SAMPLE=/opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive
doca() {
	# In WSL the GPU driver libraries live on the host in /usr/lib/wsl/lib. Without them the
	# container has nvidia-smi but no CUDA device, which hides the real (hardware) blocker.
	local wsl=() pre=""
	if [ -d /usr/lib/wsl/lib ]; then
		wsl=(-v /usr/lib/wsl:/usr/lib/wsl:ro)
		pre="export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH; "
	fi
	sudo docker run --rm --gpus all --privileged --net=host "${wsl[@]}" \
		-v "$ROOT/step2/doca_build:/build" "$DOCA_IMG" bash -c "$pre$1"
}
check "Step 4: pull DOCA container $DOCA_IMG" sudo docker pull "$DOCA_IMG"
check "Step 4: DOCA tools run" doca 'doca_caps --version && doca_caps --list-libs'
blocker "Step 4: DOCA finds a supported network card" doca \
	'doca_caps --list-devs 2>&1 | tee /tmp/devs; grep -q -i mlx5 /tmp/devs'
check "Step 4: build DOCA GPUNetIO sample" doca \
	"cd $SAMPLE && rm -rf /build/rx && meson setup /build/rx && ninja -C /build/rx"
# The sample loops forever once it starts, so a timeout (exit 124) means it worked.
blocker "Step 4: run DOCA GPUNetIO sample (NIC $NIC_PCI, GPU $GPU_PCI)" doca \
	"timeout 20 /build/rx/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0; [ \$? -eq 124 ]"

# ---------------------------------------------------------------- summary
section "Summary"
printf '%s\n' "${RESULTS[@]}"
echo
echo "Step 1 results (ms per batch):"
cat "$ROOT/step1/results.csv"
echo
echo "Step 2 results:"
cat "$ROOT/step2/step2_results.txt"
echo
echo "DOCA GPUNetIO requirements on this machine:"
bash "$ROOT/step2/check_doca_requirements.sh" | grep -E '^(OK|MISSING)'
echo
echo "Full log: $LOG"
