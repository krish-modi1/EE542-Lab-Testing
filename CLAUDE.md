# Context for Claude Code

## What this repo is

EE542 at USC (Prof. Young Cho) wants a lab where students receive images over the network with NVIDIA DOCA GPUNetIO and classify them on the GPU with a TensorRT CNN, spread across several GPU machines. This repo tests how much of that runs without DOCA hardware. Prof. Cho asked for:

1. A TensorRT CNN that takes its input from GPU buffers (step1/)
2. Simulated DOCA packet handling with PCAP and DPDK (step2/)
3. Real DOCA, which needs NVIDIA LaunchPad or similar hardware (step2/build_gpunetio.sh)
4. A test of the free DOCA container from NGC (step 4 in run_all.sh)

`run_all.sh` runs all of it and writes one log to `logs/run_all_<date>.log`. That log goes to Prof. Cho as proof of what runs, what was validated, and where DOCA's hardware requirements stop us. README.md explains every step.

## Known facts (checked, do not re-research)

- GPUNetIO needs a ConnectX-6 Dx or newer, or BlueField, with GPUDirect RDMA to the GPU, kernel 6.2+, and the open NVIDIA driver. DOCA has no software NIC emulator. On a machine without that card, steps 3 and 4 must end BLOCKED. That is the expected result, not a bug.
- Previous results on a Vast.ai RTX 3060 VM: Step 1 TensorRT FP16 is 4.6x to 5.1x faster than PyTorch with 100% top-1 agreement. Step 2 got 1,728 of 1,728 packets, 16 of 16 images byte-exact, prediction match 1.000. Step 3 built and then failed with `open_ib_device(): Failed to get RDMA devices list`.
- TensorRT 11 removed BuilderFlag.FP16. FP16 comes from the FP16 ONNX file and a strongly typed network.
- The DOCA image tag `devel-cuda13.0.0-3.5.0-devel-host` needs NVIDIA driver 580+. With an older driver, pick a CUDA 12 `-devel-host` tag from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/doca/containers/doca and pass it as `DOCA_IMG=...`.

## The loop

Machine: Krish's laptop with an RTX 3060 Laptop GPU.

1. Check the platform first. If this is Windows, work inside WSL2 Ubuntu. Docker must reach the GPU (`docker run --rm --gpus all ubuntu nvidia-smi -L`); with Docker Desktop, enable WSL integration.
2. Run `bash run_all.sh` (use `SKIP_SETUP=1` after the first full run). Read the newest log in `logs/`.
3. For every FAIL line in the summary, find the cause in the log, fix it with the smallest change, and run again.
4. Stop when every line is PASS or BLOCKED. Then check that each BLOCKED step failed because of the missing NVIDIA network card or RDMA device, not because of a bug in our code or setup. Quote the error line that proves it.
5. Finish with a short report: what you changed and why, the final summary block, and the path of the final log.

## Rules

- Never fake a result. Do not edit expected outputs, loosen a check, or catch an error so a step looks like PASS. If something cannot work on this laptop, leave it FAIL or BLOCKED and explain why.
- Keep the code simple (KISS). Fix the cause, keep the style of the existing scripts, and add no new tools or frameworks.
- If a step 4 path or command from NVIDIA's docs is wrong for this image version, look inside the container (`docker run --rm -it $DOCA_IMG bash`) and use the real path.
- Ask before anything that needs sudo beyond apt, docker, and running dpdk_rx. Ask before changing drivers, the kernel, or Windows settings.
- No secrets in files, logs, or commits. If NGC asks for a login, stop and ask Krish to run `docker login nvcr.io` himself.
- Commit fixes with clear messages. Ask before pushing. Commit the final log, step1/results.csv, and step2/step2_results.txt.
