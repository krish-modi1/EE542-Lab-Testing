#!/usr/bin/env bash
# Unattended setup of one CloudLab d7525 node (A30 GPU + ConnectX-6 Dx). Runs as root.
# profile.py starts it on every boot:  sudo bash /local/repository/cloudlab/boot.sh <receiver|sender>
#
#   1 hw        hardware facts -> /mydata/logs/hw_<hostname>.env
#   2 driver    NVIDIA open driver (580 branch) + CUDA 13.0 toolkit; reboots once if needed
#   3 gpunetio  host setup from NVIDIA's GPUNetIO guide: dmabuf, GDRCopy, hugepages, ACS, ...
#   4 docker    Docker + NVIDIA container toolkit, DOCA image, doca_caps must see an mlx5 device
#   5 sample    build gpunetio_simple_receive in the container and start it once (no traffic)
#   6 baseline  receiver only: run_all.sh Steps 1-2 on the A30
#
# Safe to run again. Stages 2 and 6 leave a marker in /mydata/boot_state and are skipped
# afterwards; the others check before they install and reapply runtime settings (hugepages,
# gdrdrv, ACS, persistence mode), which are lost on reboot.
# Logs: /mydata/logs/boot_<hostname>.log; one PASS/FAIL/INFO line per check in SUMMARY_<hostname>.txt
#
# DRY_RUN=1 bash boot.sh receiver    run stage 1 (read-only), print the commands of the others
set -o pipefail
ROLE=${1:-receiver}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DOCA_IMG=${DOCA_IMG:-nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host}
SAMPLE=/opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive
HOST=$(hostname -s)
DATA=/mydata
[ -n "$DRY_RUN" ] && DATA=${DRY_DATA:-/tmp/ee542_dryrun}
LOGS=$DATA/logs STATE=$DATA/boot_state
APT="apt-get -o DPkg::Lock::Timeout=900 -y"
export DEBIAN_FRONTEND=noninteractive PATH=/usr/local/cuda/bin:$PATH

# CloudLab mounts the blockstore during boot; wait for it (10 min) before writing anything.
if [ -z "$DRY_RUN" ]; then
	for _ in $(seq 120); do mountpoint -q /mydata && break; sleep 5; done
	if ! mountpoint -q /mydata; then
		echo "$(date) /mydata is not mounted after 10 minutes, stopping" >> /var/log/ee542_boot.log
		exit 1
	fi
fi
mkdir -p "$LOGS" "$STATE" && chmod 1777 "$LOGS"
exec 9>"$STATE/lock"
flock -n 9 || { echo "boot.sh is already running"; exit 0; }
SUMMARY=$LOGS/SUMMARY_$HOST.txt
exec > >(tee -a "$LOGS/boot_$HOST.log") 2>&1

result() { echo "$1  $2" | tee -a "$SUMMARY"; }
# check TEXT CMD...  PASS if CMD succeeds, else FAIL
check() { local text=$1; shift; if "$@"; then result PASS "$text"; else result FAIL "$text"; return 1; fi; }
vge() { [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]; }   # version $1 >= $2
# PCIe bridges between the CPU and device $1 (bus:dev.fn)
pci_bridges() { readlink -f "/sys/bus/pci/devices/0000:$1" | tr / '\n' | grep -E '^[0-9a-f]{4}:' | head -n -1; }

# doca CMD  run CMD in the DOCA container with the NIC, GPU, hugepages and host GDRCopy library
doca() {
	docker run --rm --privileged --net=host --gpus all \
		-v /dev/infiniband:/dev/infiniband -v /dev/hugepages:/dev/hugepages \
		-v /opt/mellanox/gdrcopy/src:/opt/gdrcopy:ro -v "$DATA/doca_build:/build" \
		"$DOCA_IMG" bash -c "export LD_LIBRARY_PATH=/opt/gdrcopy:\$LD_LIBRARY_PATH; $1"
}

# stage N NAME [once]  run stage_NAME and record PASS/FAIL with its run time.
# With "once", a stage that passed on an earlier boot is skipped.
stage() {
	local label="stage $1 $2" fn=stage_$2 once=$3 start=$SECONDS
	echo; echo "================ $label ================ start $(date)"
	if [ -n "$once" ] && [ -f "$STATE/$2.done" ]; then
		result PASS "$label (done on an earlier boot)"
	elif [ -n "$DRY_RUN" ] && [ "$2" != hw ]; then
		declare -f "$fn"; result DRYRUN "$label"
	elif "$fn"; then
		result PASS "$label ($((SECONDS - start)) s)"
		[ -z "$once" ] || touch "$STATE/$2.done"
	else
		result FAIL "$label ($((SECONDS - start)) s)"
	fi
	echo "================ $label ================ end $(date)"
}

stage_hw() {
	modprobe mlx5_ib 2>/dev/null   # RDMA device for the ConnectX port, usually loaded already
	echo "+ lspci | grep -i -E 'nvidia|mellanox'"; lspci | grep -i -E 'nvidia|mellanox'
	echo "+ uname -r"; uname -r
	echo "+ ibv_devices"; ibv_devices
	echo "+ ip -br addr"; ip -br addr
	# The experiment interface holds 10.10.1.x (a VLAN shows up as name@parent).
	local line ifc phys nic_pci="" ib_dev="" mac="" gpu_pci rc=0
	line=$(ip -br -4 addr 2>/dev/null | awk '/ 10\.10\.1\./ {print $1; exit}')
	ifc=${line%@*} phys=${line#*@}
	if [ -n "$phys" ] && [ -e "/sys/class/net/$phys/device" ]; then
		nic_pci=$(basename "$(readlink -f "/sys/class/net/$phys/device")")
		nic_pci=${nic_pci#0000:}
		ib_dev=$(ls "/sys/class/net/$phys/device/infiniband" 2>/dev/null | head -n 1)
		mac=$(cat "/sys/class/net/$ifc/address")
	fi
	gpu_pci=$(lspci -d 10de: 2>/dev/null | awk '/3D|VGA/ {print $1; exit}')
	printf 'IFACE=%s\nNIC_PCI=%s\nIB_DEV=%s\nMAC=%s\nGPU_PCI=%s\n' \
		"$ifc" "$nic_pci" "$ib_dev" "$mac" "$gpu_pci" | tee "$LOGS/hw_$HOST.env"
	[ "$ifc" = "$phys" ] || result INFO "hw: $ifc is a VLAN on $phys; the sample must see tagged frames"
	[ -n "$ifc" ] || { result FAIL "hw: no interface with a 10.10.1.x address"; rc=1; }
	[ -n "$ib_dev" ] || { result FAIL "hw: no mlx5 RDMA device behind the 10.10.1.x interface"; rc=1; }
	[ -n "$gpu_pci" ] || { result FAIL "hw: no NVIDIA GPU in lspci"; rc=1; }
	return $rc
}

stage_driver() {
	# https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
	# The DOCA image (CUDA 13.0) needs driver 580+; pin the 580 branch before installing.
	$APT update && $APT install "linux-headers-$(uname -r)" build-essential pkg-config git wget curl \
		pciutils rdma-core ibverbs-utils libibverbs-dev mstflint python3-scapy python3-numpy || return 1
	if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
		wget -q --timeout=60 -O /tmp/cuda-keyring.deb \
			https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb &&
			dpkg -i /tmp/cuda-keyring.deb && $APT update || return 1
	fi
	$APT install nvidia-driver-pinning-580 && $APT install nvidia-open cuda-toolkit-13-0 || return 1
	if ! nvidia-smi; then
		if [ ! -f "$STATE/driver.rebooted" ]; then
			touch "$STATE/driver.rebooted"
			result INFO "driver: nvidia-smi fails after install, rebooting once ($(date))"
			sync; reboot; exit 0
		fi
		return 1
	fi
	nvcc --version | tail -n 2
	grep -q "Open Kernel Module" /proc/driver/nvidia/version
}

stage_gpunetio() {
	# Host setup from NVIDIA's GPUNetIO guide for DOCA 3.5.0 (the container's version):
	#   https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-installation-and-setup
	local rc=0 v b p bad=0
	# GPU memory mapping: dmabuf is the default with the nvidia-open driver. It needs kernel 6.2+,
	# libibverbs 1.14.44+ and the open driver. nvidia-peermem is only the fallback.
	v=$(uname -r | cut -d- -f1)
	check "gpunetio: kernel $v >= 6.2 (dmabuf)" vge "$v" 6.2 || rc=1
	v=$(basename "$(readlink -f /usr/lib/x86_64-linux-gnu/libibverbs.so.1)"); v=${v#libibverbs.so.}
	check "gpunetio: libibverbs $v >= 1.14.44 (dmabuf)" vge "$v" 1.14.44 || rc=1
	check "gpunetio: NVIDIA open kernel module (dmabuf)" grep -q "Open Kernel Module" /proc/driver/nvidia/version || rc=1
	if modprobe nvidia-peermem; then result INFO "gpunetio: nvidia-peermem loaded (fallback only)"
	else result INFO "gpunetio: nvidia-peermem did not load (only needed if dmabuf fails)"; fi

	# GDRCopy with the gdrdrv module, same path as the guide. insmod is lost on reboot.
	if [ ! -f /opt/mellanox/gdrcopy/src/gdrdrv/gdrdrv.ko ]; then
		$APT install check kmod
		rm -rf /opt/mellanox/gdrcopy
		timeout 600 git clone --depth 1 -b v2.5.2 https://github.com/NVIDIA/gdrcopy.git /opt/mellanox/gdrcopy &&
			make -C /opt/mellanox/gdrcopy CUDA=/usr/local/cuda
	fi
	lsmod | grep -q gdrdrv || (cd /opt/mellanox/gdrcopy && ./insmod.sh)
	check "gpunetio: GDRCopy gdrdrv module loaded" bash -c 'lsmod | grep -q gdrdrv' || rc=1

	# Hugepages: not in the GPUNetIO guide, but the Ethernet samples run DOCA Flow, which uses DPDK.
	echo 2048 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
	check "gpunetio: 2048 x 2 MB hugepages" bash -c 'grep -q "HugePages_Total: *2048$" /proc/meminfo' || rc=1

	check "gpunetio: GPU persistence mode on" nvidia-smi -pm 1 || rc=1

	# BAR1 must be large enough to map GPU buffers to the NIC (small on RTX cards: 256 MB).
	v=$(nvidia-smi -q -d MEMORY | awk '/BAR1/ {f=1} f && /Total/ {print $3; exit}')
	check "gpunetio: GPU BAR1 ${v:-?} MiB > 256 MiB" [ "${v:-0}" -gt 256 ] || rc=1

	# Topology: PIX/PXB is best, PHB is fine, NODE/SYS is slower. A hardware fact, so INFO.
	nvidia-smi topo -m
	v=$(nvidia-smi topo -m | awk -v dev="$IB_DEV" 'NR == 1 {for (i = 1; i <= NF; i++) col[$i] = i + 1}
		$1 == "GPU0" {for (k in col) row[k] = $(col[k])}
		$1 ~ /^NIC[0-9]+:$/ && $2 == dev {print row[substr($1, 1, length($1) - 1)]}')
	result INFO "gpunetio: GPU0 to $IB_DEV topology is ${v:-unknown} (guide: PIX/PXB best, avoid NODE/SYS)"

	# ACS off on the PCIe bridges above the GPU and the NIC. The guide prefers the BIOS, which we
	# cannot reach on CloudLab, so use its setpci alternative. Lost on reboot.
	for b in $( (pci_bridges "$GPU_PCI"; pci_bridges "$NIC_PCI") | sort -u); do
		lspci -s "$b" -vvv | grep -q ACSCtl || continue
		setpci -s "$b" ECAP_ACS+0x6.w=0000
		echo "$b $(lspci -s "$b" -vvv | grep ACSCtl)"
		lspci -s "$b" -vvv | grep ACSCtl | grep -q '+' && bad=1
	done
	check "gpunetio: ACS disabled on the GPU and NIC bridges" [ $bad = 0 ] || rc=1

	# The guide turns the IOMMU off only "if the application receives no packets". Not changed here.
	cat /proc/cmdline
	if [ -n "$(ls /sys/class/iommu 2>/dev/null)" ]; then
		result INFO "gpunetio: IOMMU is on; if the sample gets no packets, add amd_iommu=off iommu=off to GRUB and reboot"
	else
		result INFO "gpunetio: IOMMU is off"
	fi

	# NIC firmware: the guide sets KEEP_ETH_LINK_UP_P<n>=1 and KEEP_IB_LINK_UP_P<n>=0 with mlxconfig and
	# then needs a cold power cycle. Firmware settings outlive our reservation on a shared testbed,
	# so only query them (mstconfig is mlxconfig from Ubuntu's mstflint package).
	p=$((${NIC_PCI##*.} + 1))
	mstconfig -d "$NIC_PCI" q | tee "$LOGS/mstconfig_$HOST.txt" | grep -E 'LINK_TYPE|KEEP_.*LINK_UP'
	if ! grep -q "KEEP_ETH_LINK_UP_P$p" "$LOGS/mstconfig_$HOST.txt"; then
		result INFO "gpunetio: firmware does not expose KEEP_ETH_LINK_UP_P$p (see mstconfig_$HOST.txt)"
	elif grep -q -E "KEEP_ETH_LINK_UP_P$p +True" "$LOGS/mstconfig_$HOST.txt" &&
		! grep -q -E "KEEP_IB_LINK_UP_P$p +True" "$LOGS/mstconfig_$HOST.txt"; then
		result PASS "gpunetio: NIC firmware KEEP_ETH_LINK_UP_P$p=1, KEEP_IB_LINK_UP_P$p=0"
	else
		result FAIL "gpunetio: NIC firmware differs from the guide (KEEP_*_LINK_UP_P$p); changing it needs mlxconfig + cold power cycle, not done on a shared node"
		rc=1
	fi
	return $rc
}

stage_docker() {
	# Docker data on /mydata: the root disk is 64 GB and the DOCA image alone is about 22 GB.
	mkdir -p /etc/docker "$DATA/docker"
	[ -f /etc/docker/daemon.json ] || echo '{ "data-root": "/mydata/docker" }' > /etc/docker/daemon.json
	command -v docker >/dev/null || $APT install docker.io || return 1
	# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
	if ! command -v nvidia-ctk >/dev/null; then
		curl -fsSL --max-time 60 https://nvidia.github.io/libnvidia-container/gpgkey |
			gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
		curl -fsSL --max-time 60 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
			sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
				> /etc/apt/sources.list.d/nvidia-container-toolkit.list
		$APT update && $APT install nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker || return 1
	fi
	# Docker may have started before /mydata was mounted; restart it so it uses /mydata/docker.
	systemctl restart docker
	check "docker: data-root is /mydata/docker" bash -c 'docker info 2>/dev/null | grep -q "Docker Root Dir: /mydata/docker"' || return 1
	check "docker: GPU visible in a container" docker run --rm --gpus all ubuntu nvidia-smi -L || return 1
	check "docker: pull $DOCA_IMG" timeout 3600 docker pull -q "$DOCA_IMG" || return 1
	doca 'doca_caps --list-devs' 2>&1 | tee "$LOGS/doca_caps_$HOST.txt"
	check "docker: doca_caps --list-devs shows an mlx5 device" grep -q -i mlx5 "$LOGS/doca_caps_$HOST.txt"
}

stage_sample() {
	# Same build as run_all.sh Step 4, with the output on /mydata.
	[ -x "$DATA/doca_build/rx/doca_gpunetio_simple_receive" ] ||
		doca "cd $SAMPLE && rm -rf /build/rx && meson setup /build/rx && ninja -C /build/rx"
	check "sample: gpunetio_simple_receive built" test -x "$DATA/doca_build/rx/doca_gpunetio_simple_receive" || return 1
	# Start it with no traffic. It prints the packet total only after NIC and GPU setup succeeded
	# and it exited on SIGINT, so that line proves the receive path starts on this node.
	doca "timeout -s INT 30 /build/rx/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0" 2>&1 |
		tee "$LOGS/sample_start_$HOST.log"
	check "sample: starts on NIC $NIC_PCI and GPU $GPU_PCI" grep -q "Total number of received packets" "$LOGS/sample_start_$HOST.log"
}

stage_baseline() {
	# run_all.sh Steps 1-2 (TensorRT, DPDK from pcap) on the A30. Its own log has the details.
	VENV=$DATA/venv SKIP_DOCA=1 bash "$REPO/run_all.sh" > /dev/null 2>&1
	local log
	log=$(ls -t "$REPO"/logs/run_all_*.log | head -n 1)
	cp "$log" "$LOGS/"
	echo "run_all log: $LOGS/$(basename "$log")"
	sed -n '/=== Summary ===/,/^$/p' "$log"
	grep -q '=== Summary ===' "$log" && ! grep -q '^FAIL' "$log"
}

[ -n "$DRY_RUN" ] || systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer
echo "=== boot.sh $ROLE on $HOST started $(date) ===" | tee -a "$SUMMARY"
stage 1 hw
. "$LOGS/hw_$HOST.env"
stage 2 driver once
stage 3 gpunetio
stage 4 docker
stage 5 sample
if [ "$ROLE" = receiver ]; then stage 6 baseline once; fi
echo "=== boot.sh finished $(date), $((SECONDS / 60)) min ===" | tee -a "$SUMMARY"
