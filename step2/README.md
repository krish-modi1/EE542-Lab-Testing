# Step 2: Simulated packet delivery into the Step 1 inference pipeline

Images travel as UDP packets in a PCAP file. DPDK's pcap driver replays the file with no
physical NIC, a DPDK program reassembles the frames, and the Step 1 TensorRT engine runs on
them in GPU memory. The last part builds NVIDIA's open-source GPUNetIO and shows that running
it needs an RDMA-capable NVIDIA NIC (the Step 3 blocker).

Tested without a GPU: `make_pcap.py`, `rx_python.py`, and `dpdk_rx` (DPDK 23.11, Ubuntu 24.04)
received all 1,728 packets and rebuilt all 16 frames byte for byte, in order and shuffled.
`infer_rx.py` and `build_gpunetio.sh` need the GPU VM and have not been run yet.

## Run order (on the GPU VM, from `step2/`, with the Step 1 venv active)
```bash
sudo apt-get install -y dpdk dpdk-dev libdpdk-dev libpcap-dev pkg-config build-essential
pip install scapy
mkdir -p logs

python3 make_pcap.py                       # 2a. images.pcap + frames.npy (16 images, 1728 packets)
make                                       # 2b. build the DPDK receiver
./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
  --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null' -- frames_rx.bin | tee logs/dpdk_rx.log
#   DPDK won't build or run?  python3 rx_python.py   (same output file)
python3 infer_rx.py                        # 2c. GPU preprocessing + TensorRT -> step2_results.txt
bash build_gpunetio.sh 86                  # 2d. GPUNetIO build + expected failed run -> logs/
```
Use `python3 make_pcap.py --shuffle` to test out-of-order packets.

## Packet format
Each 224x224x3 image (150,528 bytes) is split into 108 UDP payloads of up to 1,400 bytes.
Every payload starts with a 24-byte header: magic, frame id, segment number, segment count,
byte offset, length, padding, frame size (little-endian).

## What this shows, and what it does not
- The packet parsing and reassembly logic works, and the received frames are byte-identical.
- The frames reach GPU memory through a host copy that `infer_rx.py` times. DOCA GPUNetIO
  would remove that copy by letting the NIC write payloads straight into GPU memory.
- The receiver uses DPDK, not DOCA Flow or GPUNetIO. Running those needs a ConnectX-6 Dx or
  newer, or BlueField, NIC; `build_gpunetio.sh` records that failure as evidence.

## Evidence to keep
`logs/dpdk_rx.log`, `step2_results.txt`, `logs/rdma_devices.log`,
`logs/gpunetio_build.log`, `logs/gpunetio_run.log`
