#!/usr/bin/env bash
# Two-node DOCA GPUNetIO test on CloudLab, run by hand after boot.sh has finished.
#
#   node0:  bash gpunetio_test.sh receive   run NVIDIA's gpunetio_simple_receive for DURATION s (default 120)
#   node1:  bash gpunetio_test.sh send      send Step 2's 1,728 image packets (UDP) to 10.10.1.1
#
# Start receive first, then send within DURATION seconds. The sample's flow rule takes any
# IPv4/UDP packet (any port, any MAC) and counts it on the GPU; it prints the total when it
# exits on SIGINT. PASS = that total is above zero.
# The sender needs node0's MAC. It pings 10.10.1.1 and reads the ARP table; if that fails
# (node0 may not answer ARP while the sample owns the port), copy the MAC from the receive
# log and run DST_MAC=<mac> bash gpunetio_test.sh send.
# Log: /mydata/logs/gpunetio_test_<hostname>.log
set -o pipefail
MODE=$1
DURATION=${DURATION:-120}
SRC_IP=10.10.1.2 DST_IP=10.10.1.1
DOCA_IMG=${DOCA_IMG:-nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host}
REPO=$(cd "$(dirname "$0")/.." && pwd)
HOST=$(hostname -s)
LOG=/mydata/logs/gpunetio_test_$HOST.log
. "/mydata/logs/hw_$HOST.env" || { echo "run boot.sh first (no hw_$HOST.env)"; exit 1; }
exec > >(tee -a "$LOG") 2>&1
echo "================ gpunetio_test.sh $MODE on $HOST, $(date) ================"
cat "/mydata/logs/hw_$HOST.env"

receive() {
	echo "This node: $IFACE $DST_IP MAC $MAC (use DST_MAC=$MAC on the sender if ARP fails)"
	# Same container options as boot.sh: NIC, GPU, hugepages, host GDRCopy library.
	sudo docker run --rm --privileged --net=host --gpus all \
		-v /dev/infiniband:/dev/infiniband -v /dev/hugepages:/dev/hugepages \
		-v /opt/mellanox/gdrcopy/src:/opt/gdrcopy:ro -v /mydata/doca_build:/build "$DOCA_IMG" \
		bash -c "export LD_LIBRARY_PATH=/opt/gdrcopy:\$LD_LIBRARY_PATH
			timeout -s INT $DURATION /build/rx/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0" \
		2>&1 | tee /tmp/gpunetio_rx.txt
	local n
	n=$(sed -n 's/.*Total number of received packets: \([0-9]*\).*/\1/p' /tmp/gpunetio_rx.txt)
	if [ "${n:-0}" -gt 0 ]; then
		echo "PASS  GPUNetIO received $n packets on the GPU (the sender sends 1728 per round)"
	else
		echo "FAIL  GPUNetIO received ${n:-no total printed} packets"
		return 1
	fi
}

send() {
	local dst_mac=$DST_MAC
	if [ -z "$dst_mac" ]; then
		ping -c 3 -W 1 "$DST_IP"
		dst_mac=$(ip neigh show "$DST_IP" | awk '{for (i = 1; i < NF; i++) if ($i == "lladdr") print $(i + 1)}')
	fi
	[ -n "$dst_mac" ] || { echo "FAIL  no MAC for $DST_IP; set DST_MAC (printed by the receiver)"; return 1; }
	# Reuse Step 2's packets (24-byte header + image chunk, UDP port 5000), readdressed to node0.
	mkdir -p /tmp/gpunetio_test && cd /tmp/gpunetio_test || return 1
	python3 "$REPO/step2/make_pcap.py" || return 1
	sudo python3 - "$IFACE" "$MAC" "$SRC_IP" "$dst_mac" "$DST_IP" <<'EOF'
import sys
from scapy.all import IP, UDP, Ether, rdpcap, sendp
iface, src_mac, src_ip, dst_mac, dst_ip = sys.argv[1:]
pkts = rdpcap("images.pcap")
for p in pkts:
    p[Ether].src, p[Ether].dst = src_mac, dst_mac
    p[IP].src, p[IP].dst = src_ip, dst_ip
    del p[IP].len, p[IP].chksum, p[UDP].len, p[UDP].chksum
sendp(pkts, iface=iface, verbose=False)
print(f"sent {len(pkts)} packets from {iface} ({src_mac}) to {dst_ip} ({dst_mac}) UDP port 5000")
EOF
}

case $MODE in
	receive) receive ;;
	send) send ;;
	*) echo "usage: bash gpunetio_test.sh receive|send"; exit 1 ;;
esac
