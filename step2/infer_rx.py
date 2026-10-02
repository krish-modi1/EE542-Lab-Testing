"""Step 2c: move the reassembled frames into GPU memory, preprocess them on the GPU, and run
the Step 1 TensorRT engine. Checks that:
  1. every received frame is byte-identical to the original image, and
  2. predictions on received frames match predictions on the originals.

The host-to-GPU copy timed here is the hop that DOCA GPUNetIO would remove: with GPUNetIO
the NIC writes packet payloads directly into GPU memory.

Usage: python3 infer_rx.py [--engine ../step1/resnet18_fp16.engine] [--rx frames_rx.bin]
Writes step2_results.txt.
"""
import argparse
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "step1"))
from gpu_buffer_runner import load_engine, trt_infer_fn  # noqa: E402

H = W = 224
FRAME_BYTES = H * W * 3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_rx(path):
    raw = open(path, "rb").read()
    rec = 4 + FRAME_BYTES
    if len(raw) % rec:
        raise SystemExit(f"{path}: size {len(raw)} is not a multiple of {rec}")
    ids, frames = [], []
    for i in range(0, len(raw), rec):
        ids.append(struct.unpack_from("<I", raw, i)[0])
        frames.append(np.frombuffer(raw, np.uint8, FRAME_BYTES, i + 4).reshape(H, W, 3))
    return ids, np.stack(frames)


def preprocess_gpu(u8):
    """uint8 NHWC on GPU -> normalized float32 NCHW on GPU."""
    x = u8.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def predict(engine, u8_gpu):
    run, stream = trt_infer_fn(engine, u8_gpu.shape[0])
    x = preprocess_gpu(u8_gpu)
    stream.wait_stream(torch.cuda.current_stream())  # TensorRT stream waits for preprocessing
    out = run(x)
    stream.synchronize()
    return out.float().argmax(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="../step1/resnet18_fp16.engine")
    ap.add_argument("--rx", default="frames_rx.bin")
    ap.add_argument("--orig", default="frames.npy")
    args = ap.parse_args()

    orig = np.load(args.orig)
    ids, rx = read_rx(args.rx)
    if len(ids) > 32:
        raise SystemExit("more than 32 frames; the engine profile allows batch <= 32")

    # 1. Byte-exact reassembly check
    missing = sorted(set(range(len(orig))) - set(ids))
    exact = [bool(np.array_equal(rx[k], orig[fid])) for k, fid in enumerate(ids)]

    # 2. Host -> GPU copy (the hop GPUNetIO removes), then GPU preprocessing + inference
    engine = load_engine(args.engine)
    pinned = torch.from_numpy(rx).pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rx_gpu = pinned.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    t_copy = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    pred_rx = predict(engine, rx_gpu)
    t_infer = (time.perf_counter() - t0) * 1e3

    orig_gpu = torch.from_numpy(orig[ids]).to("cuda")
    pred_orig = predict(engine, orig_gpu)
    match = (pred_rx == pred_orig).float().mean().item()

    lines = [
        f"GPU: {torch.cuda.get_device_name(0)}",
        f"engine: {args.engine}",
        f"frames received: {len(ids)} of {len(orig)} (missing: {missing or 'none'})",
        f"byte-exact frames: {sum(exact)} of {len(ids)}",
        f"prediction match (received vs original): {match:.3f}",
        f"host->GPU copy of {len(ids)} frames ({rx.nbytes / 1e6:.2f} MB): {t_copy:.3f} ms",
        f"GPU preprocess + inference (first call, includes context setup): {t_infer:.3f} ms",
        f"predicted classes (first 8): {pred_rx[:8].tolist()}",
    ]
    print("\n".join(lines))
    with open("step2_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote step2_results.txt")


if __name__ == "__main__":
    main()
