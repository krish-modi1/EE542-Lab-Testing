# EE542 DOCA lab testing

This repo checks how much of a planned EE542 lab can run on an ordinary cloud GPU. The lab idea is to receive images over the network with NVIDIA DOCA GPUNetIO and classify them on the GPU with a CNN.

GPUNetIO lets the network card write packets straight into GPU memory, so the CPU never copies the data. It needs a special NVIDIA network card (ConnectX-6 Dx or newer, or BlueField). This repo tests every part of the pipeline that works without that card, then shows exactly where the missing card stops it.

| Step | What it tests | Result on an RTX 3060 |
|---|---|---|
| 1 | TensorRT CNN reading its input from GPU memory | Works. FP16 is about 5x faster than PyTorch with the same predictions |
| 2 | Images sent as UDP packets, rebuilt with DPDK, then classified | Works. 1,728 packets, 16 of 16 images rebuilt exactly, predictions match |
| 3 | NVIDIA's open-source DOCA GPUNetIO | Builds, then stops: no RDMA network card on the machine |
| 4 | NVIDIA's official DOCA container from NGC | Not run yet (see `run_all.sh`) |

## Repo layout

```
run_all.sh              run every step below and write one shareable log
CLAUDE.md               context for running and fixing this with Claude Code
profile.py              CloudLab profile: two d7525 nodes (see "CloudLab" below)
cloudlab/
  boot.sh               unattended node setup, runs on every boot
  gpunetio_test.sh      two-node GPUNetIO send/receive test, run by hand
step1/
  setup_check.sh        print GPU, driver, CUDA and package versions
  export_onnx.py        ResNet-18 -> resnet18_fp32.onnx and resnet18_fp16.onnx
  build_engine.py       ONNX -> TensorRT engine (run once for fp32, once for fp16)
  gpu_buffer_runner.py  benchmark TensorRT vs PyTorch, writes results.csv
  results.csv           our results on an RTX 3060
step2/
  make_pcap.py          16 images -> 1,728 UDP packets in images.pcap
  dpdk_rx.c, Makefile   DPDK program that reads images.pcap and rebuilds the images
  infer_rx.py           classify the rebuilt images with the Step 1 engine
  check_doca_requirements.sh  Step 3: compare this machine with what GPUNetIO needs
  build_gpunetio.sh     Step 3: build NVIDIA's open-source GPUNetIO and run one example
  logs/                 our DPDK, GPUNetIO build and GPUNetIO run logs
  step2_results.txt     our Step 2 results
```

## What you need

Any Linux machine with an NVIDIA GPU that has tensor cores (RTX 30xx or 40xx, T4, L4, A10 or newer) and root access. We used a Vast.ai VM: RTX 3060 12 GB, Ubuntu 22.04, driver 580, CUDA 12.6, TensorRT 11.3, PyTorch 2.14. A laptop GPU also works for Steps 1 and 2.

## Run everything at once

```bash
bash run_all.sh                 # first run: installs everything, then runs all steps
SKIP_SETUP=1 bash run_all.sh    # later runs
```

The script installs the packages, runs Steps 1 to 4 and writes one log to `logs/run_all_<date>.log`. The log ends with one line per check: PASS, FAIL or BLOCKED. BLOCKED means the step needs an NVIDIA RDMA network card that the machine does not have. The sections below explain each step and how to run it by hand.

## Setup (once)

```bash
# CUDA compiler, needed for Step 3 (skip if `nvcc --version` already works)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
export PATH=/usr/local/cuda/bin:$PATH

# DPDK and RDMA libraries for Steps 2 and 3
sudo apt-get install -y dpdk dpdk-dev libdpdk-dev libpcap-dev pkg-config build-essential \
  rdma-core libibverbs-dev ibverbs-utils

# Python packages, in a virtual environment
sudo apt-get install -y python3-venv
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch torchvision onnx onnxscript tensorrt scapy
```

Run `source ~/venv/bin/activate` again in every new terminal. If you forget, Python reports `No module named 'torch'`.

## Step 1: TensorRT inference from GPU memory

With GPUNetIO, the image data is already in GPU memory when inference starts. So the inference code has to accept a GPU memory address instead of a normal array. Step 1 builds that part and measures it.

The model is ResNet-18, a pretrained image classifier from torchvision. TensorRT converts it into an engine that is optimised for your specific GPU.

```bash
cd step1
bash setup_check.sh
python3 export_onnx.py
python3 build_engine.py fp32
python3 build_engine.py fp16
python3 gpu_buffer_runner.py
```

`gpu_buffer_runner.py` feeds random input to PyTorch and to both engines at batch sizes 1, 8 and 32. Random input is enough here, because we only measure speed and check that TensorRT gives the same answer as PyTorch. The main idea is in `make_runner()`: it allocates input and output buffers on the GPU and gives their raw addresses to TensorRT with `set_tensor_address`.

Our results:

| Batch | PyTorch FP32 | TensorRT FP32 | TensorRT FP16 | FP16 speedup over PyTorch |
|---|---|---|---|---|
| 1 | 2.18 ms | 2.08 ms | 0.48 ms | 4.6x |
| 8 | 5.22 ms | 3.38 ms | 1.02 ms | 5.1x |
| 32 | 17.60 ms | 12.54 ms | 3.82 ms | 4.6x |

Both engines picked the same top class as PyTorch for every image. The largest difference in raw scores was 0.036 for FP16, which is normal rounding for half precision. FP16 gains the most because the RTX 3060's tensor cores run FP16 math much faster than FP32.

## Step 2: from packets to predictions, without a network card

This step simulates the network side. A PCAP file is a recording of network traffic. DPDK, a fast packet processing library, can read a PCAP file as if packets were arriving on a real network card, so no special hardware is needed.

1. `make_pcap.py` creates 16 test images (colour gradients with noise) and splits each one into 108 UDP packets. Every packet has a small header with the image number and the position of its chunk, so the receiver can put the image back together even if packets arrive out of order.
2. `dpdk_rx` reads the packets through DPDK, checks the Ethernet, IP and UDP headers, and rebuilds the images into `frames_rx.bin`.
3. `infer_rx.py` copies the rebuilt images to the GPU, normalises them on the GPU, and classifies them with the FP16 engine from Step 1.

```bash
cd step2
mkdir -p logs
python3 make_pcap.py                 # add --shuffle to send packets out of order
make
sudo ./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
  --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null' -- frames_rx.bin | tee logs/dpdk_rx.log
python3 infer_rx.py
```

The DPDK options mean: use CPU core 0 (`-l 0`), use normal memory instead of hugepages (`--no-huge -m 512`), skip PCI devices (`--no-pci`), and create a virtual port that reads `images.pcap` (`--vdev`).

Our results: all 1,728 packets arrived, all 16 images were rebuilt byte for byte, and the received images got the same predictions as the originals. Copying the 16 images (2.4 MB) from CPU memory to the GPU took 0.50 ms. That copy is exactly what GPUNetIO removes.

The predicted classes themselves are meaningless, because the test images are synthetic patterns. The check is that received and original images get the same prediction.

## Step 3: real DOCA GPUNetIO

```bash
cd step2
bash check_doca_requirements.sh
bash build_gpunetio.sh 86            # 86 = RTX 30xx; use 89 for RTX 40xx or L4, 90 for H100
```

`check_doca_requirements.sh` prints OK or MISSING for each GPUNetIO requirement: an NVIDIA ConnectX or BlueField card, an RDMA device, kernel 6.2 or newer, the NVIDIA open kernel driver, and bare metal instead of a VM. It also prints the PCIe path between the GPU and the network card.

`build_gpunetio.sh` lists the machine's RDMA devices and network cards, builds NVIDIA's open-source GPUNetIO for your GPU, and runs one of its examples.

On our VM, the build succeeded, and the example stopped with:

```
open_ib_device(): Failed to get RDMA devices list, ibdev_list null
```

`ibv_devices` listed no RDMA devices, and `lspci` showed only a `Virtio network device`, which is the simple virtual card the cloud provider emulates. GPUNetIO needs the GPU to talk directly to an NVIDIA RDMA card, so with no such card it cannot start. The `Failed to open libgdrapi.so.2` line in the same log refers to GDRCopy, an optional speed-up library, and is not the cause.

## Step 4: NVIDIA's official DOCA container

NVIDIA publishes a free DOCA container on NGC with the full SDK, tools and samples. `run_all.sh` pulls it, asks DOCA which devices it can see (`doca_caps --list-devs`), builds NVIDIA's `gpunetio_simple_receive` sample and tries to run it. The container gives you the software, not the hardware: DOCA has no simulated network card, so on a machine without a ConnectX or BlueField card the device check and the sample run should end BLOCKED, the same as Step 3. The image needs Docker with GPU access (so it does not work inside a Vast "container" instance) and NVIDIA driver 580 or newer. For an older driver, pick a CUDA 12 `-devel-host` tag from the [NGC page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca) and run `DOCA_IMG=nvcr.io/nvidia/doca/doca:<tag> bash run_all.sh`.

## How this maps to the real lab

| Real DOCA pipeline | This repo |
|---|---|
| Packets arrive on a ConnectX or BlueField card | Packets come from `images.pcap` through DPDK |
| The card writes payloads into GPU memory (GPUNetIO) | The CPU rebuilds images, then copies them to the GPU (0.50 ms for 16 images) |
| A CUDA kernel parses headers | `dpdk_rx.c` parses headers on the CPU |
| TensorRT classifies the images in GPU memory | Same: `make_runner()` from Step 1 |

To run the left column you need two machines, each with a GPU and a ConnectX-6 Dx (or newer) or BlueField card. Standard cloud GPUs, Vast.ai instances and Lab 6's T4 and V100 instances do not have one. CloudLab's Wisconsin d7525 nodes have an A30 GPU and a ConnectX-6 Dx card and are free for research and teaching, but a faculty member has to create the project (https://docs.cloudlab.us/users.html).

## CloudLab (real DOCA hardware)

`profile.py` is a CloudLab repository-based profile (CloudLab reads it from the repo root and clones the repo to `/local/repository` on every node). It creates two bare-metal d7525 nodes at Wisconsin (NVIDIA A30 + ConnectX-6 Dx, Ubuntu 24.04) on one LAN: `node0` = receiver at 10.10.1.1, `node1` = sender at 10.10.1.2. Each node gets a `/mydata` filesystem on the rest of its disk and runs `cloudlab/boot.sh` on every boot.

`cloudlab/boot.sh` needs no one at the keyboard. It runs as root and takes about 1 to 2 hours:

| Stage | What it does |
|---|---|
| 1 hw | Records the GPU, the NIC behind 10.10.1.x, its mlx5 device and MAC in `hw_<hostname>.env` |
| 2 driver | NVIDIA open driver (580 branch) and CUDA 13.0, reboots once if the driver needs it |
| 3 gpunetio | Host setup from NVIDIA's GPUNetIO guide: dmabuf checks, GDRCopy, hugepages, persistence mode, BAR1, ACS, topology, IOMMU and NIC firmware checks |
| 4 docker | Docker (data on `/mydata/docker`), NVIDIA container toolkit, DOCA image, `doca_caps --list-devs` must show an mlx5 device |
| 5 sample | Builds `gpunetio_simple_receive` and starts it once with no traffic, to prove NIC and GPU setup work |
| 6 baseline | node0 only: `run_all.sh` Steps 1 and 2 on the A30 |

It can run twice safely: stages 2 and 6 are skipped once they pass, and the others check before installing. If something failed, run it again by hand with `sudo bash /local/repository/cloudlab/boot.sh receiver` (or `sender`). `DRY_RUN=1 bash cloudlab/boot.sh receiver` runs only stage 1 and prints the other stages without running them.

Logs on each node, in `/mydata/logs/`:

- `SUMMARY_<hostname>.txt`: one PASS, FAIL or INFO line per check. Read this first.
- `boot_<hostname>.log`: everything, with start and end times for each stage
- `hw_<hostname>.env`, `doca_caps_<hostname>.txt`, `sample_start_<hostname>.log`, `mstconfig_<hostname>.txt`, and on node0 the `run_all_*.log`

The send and receive test is run by hand:

```bash
# node1, before starting the receiver (so node1 learns node0's MAC from ARP)
ping -c 3 10.10.1.1
# node0: run the GPUNetIO receiver for 120 s (DURATION=300 for longer)
bash /local/repository/cloudlab/gpunetio_test.sh receive
# node1, while the receiver runs: send Step 2's 1,728 image packets to node0
bash /local/repository/cloudlab/gpunetio_test.sh send
```

The test passes when the receiver's own output says `Total number of received packets:` with a number above zero. If the sender cannot find node0's MAC, copy it from the receiver's output and run `DST_MAC=<mac> bash gpunetio_test.sh send`. Results go to `/mydata/logs/gpunetio_test_<hostname>.log`.

## Problems we hit

| Error | Cause and fix |
|---|---|
| `Could not get lock /var/lib/dpkg/lock-frontend` | Ubuntu's automatic updater was running. Wait for it, or run `sudo systemctl stop unattended-upgrades`, then install again |
| `No module named 'onnxscript'` | New PyTorch versions export ONNX through onnxscript. `pip install onnxscript` |
| `BuilderFlag has no attribute 'FP16'` | TensorRT 11 removed that flag. FP16 now comes from an FP16 ONNX file, which `export_onnx.py` creates |
| FP16 showed 0% agreement with PyTorch | The output was read before TensorRT finished. `make_runner()` now waits for the GPU before returning |
| `libdoca_gpunetio_host.so.4: cannot open shared object file` | The example could not find its own library. `build_gpunetio.sh` sets `LD_LIBRARY_PATH` |
