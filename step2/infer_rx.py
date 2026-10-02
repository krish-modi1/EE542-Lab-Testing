"""Classify the images rebuilt by dpdk_rx with the Step 1 FP16 engine.

Checks two things and writes them to step2_results.txt:
  1. every received image is byte-identical to the original in frames.npy
  2. the received images get the same predictions as the originals
It also times the CPU-to-GPU copy. That copy is the step DOCA GPUNetIO removes, because the
network card would write the packets straight into GPU memory.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

STEP1 = Path(__file__).resolve().parent.parent / "step1"
sys.path.insert(0, str(STEP1))
from gpu_buffer_runner import load_engine, make_runner  # noqa: E402

FRAME_BYTES = 224 * 224 * 3
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def classify(run, images_gpu):
    """uint8 images (N, 224, 224, 3) on the GPU -> predicted class per image."""
    x = images_gpu.permute(0, 3, 1, 2).float() / 255  # to (N, 3, 224, 224), range 0..1
    return run((x - MEAN) / STD).argmax(1).cpu()


originals = np.load("frames.npy")
records = np.fromfile("frames_rx.bin", dtype=np.uint8).reshape(-1, 4 + FRAME_BYTES)
ids = records[:, :4].copy().view("<u4").ravel()  # frame_id written before each image
received = np.ascontiguousarray(records[:, 4:]).reshape(-1, 224, 224, 3)

exact = sum(np.array_equal(img, originals[i]) for img, i in zip(received, ids))

# CPU -> GPU copy (the hop GPUNetIO removes)
pinned = torch.from_numpy(received).pin_memory()
torch.cuda.synchronize()
start = time.perf_counter()
received_gpu = pinned.cuda(non_blocking=True)
torch.cuda.synchronize()
copy_ms = (time.perf_counter() - start) * 1000

run = make_runner(load_engine(STEP1 / "resnet18_fp16.engine"), len(ids))
pred_received = classify(run, received_gpu)
pred_original = classify(run, torch.from_numpy(originals[ids]).cuda())
match = (pred_received == pred_original).float().mean().item()

lines = [
    f"GPU: {torch.cuda.get_device_name()}",
    f"frames received: {len(ids)} of {len(originals)}",
    f"byte-exact frames: {exact} of {len(ids)}",
    f"prediction match (received vs original): {match:.3f}",
    f"CPU->GPU copy of {len(ids)} frames ({received.nbytes / 1e6:.2f} MB): {copy_ms:.3f} ms",
]
print("\n".join(lines))
Path("step2_results.txt").write_text("\n".join(lines) + "\n")
