#!/usr/bin/env bash
# Compare this machine with what DOCA GPUNetIO needs.
# Prints OK or MISSING for each requirement and exits 1 if anything is missing.
missing=0
check() {  # check "<requirement>" "<what this machine has>" <test command...>
  local need=$1 found=$2 status=OK
  shift 2
  "$@" || { status=MISSING; missing=1; }
  printf '%-8s %-44s %s\n' "$status" "$need" "$found"
}

nic=$(lspci -d 15b3: | head -n 2 | cut -d: -f3- | paste -sd';')  # 15b3 = NVIDIA/Mellanox networking
rdma=$(ls /sys/class/infiniband 2>/dev/null | paste -sd' ')
kernel=$(uname -r)
driver=$(grep -o "Open Kernel Module" /proc/driver/nvidia/version 2>/dev/null)
virt=$(systemd-detect-virt --vm 2>/dev/null); virt=${virt:-unknown}

check "ConnectX-6 Dx or newer, or BlueField NIC" "${nic:-none; NICs: $(lspci | grep -i ethernet | cut -d: -f3- | paste -sd';')}" test -n "$nic"
check "RDMA device (mlx5_*)" "${rdma:-none}" test -n "$rdma"
check "Linux kernel 6.2 or newer (dma-buf)" "$kernel" test "$(printf '6.2\n%s\n' "${kernel%%-*}" | sort -V | head -n 1)" = 6.2
check "NVIDIA open kernel driver" "${driver:-proprietary or unknown}" test -n "$driver"
check "Not a VM (GPUNetIO in VMs is experimental)" "VM type: $virt" test "$virt" = none

echo
echo "GPU to NIC PCIe path (GPUNetIO wants them under one PCIe switch: PIX or PXB):"
nvidia-smi topo -m
exit $missing
