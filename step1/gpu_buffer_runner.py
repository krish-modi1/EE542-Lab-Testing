"""Step 1c: run the TensorRT engine on input that is ALREADY in GPU memory, and benchmark
PyTorch vs TensorRT FP32 vs TensorRT FP16.

Why GPU buffers: with DOCA GPUNetIO the NIC writes packet payloads straight into GPU memory,
so the inference code must accept a device pointer, not a host array. Here a torch CUDA tensor
stands in for that buffer (its memory comes from cudaMalloc via PyTorch's allocator), and we pass
its raw device address (tensor.data_ptr()) to TensorRT.

Written against the TensorRT 10 Python API (set_tensor_address / execute_async_v3).
Output: results.csv plus a printed table.
"""
import csv
import time

import tensorrt as trt
import torch
import torchvision

BATCHES = [1, 8, 32]
WARMUP, ITERS = 20, 200
dev = torch.device("cuda")
logger = trt.Logger(trt.Logger.WARNING)
TRT2TORCH = {trt.float32: torch.float32, trt.float16: torch.float16}


def load_engine(path):
    with open(path, "rb") as f:
        return trt.Runtime(logger).deserialize_cuda_engine(f.read())


def trt_infer_fn(engine, batch):
    """Return a closure that runs one inference on a device buffer of the given batch size."""
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", (batch, 3, 224, 224))
    # Buffer dtypes follow the engine: FP32 engine -> float32, FP16 engine -> float16.
    in_dt = TRT2TORCH[engine.get_tensor_dtype("input")]
    out_dt = TRT2TORCH[engine.get_tensor_dtype("logits")]
    # These two tensors are the "DOCA-delivered" input buffer and the output buffer.
    inp = torch.empty((batch, 3, 224, 224), device=dev, dtype=in_dt)
    out = torch.empty((batch, 1000), device=dev, dtype=out_dt)
    ctx.set_tensor_address("input", inp.data_ptr())   # raw device pointer
    ctx.set_tensor_address("logits", out.data_ptr())
    stream = torch.cuda.Stream()

    def run(src):
        # Copy and inference on the same stream so they are ordered.
        # In the real pipeline the NIC would fill `inp` instead of this copy.
        with torch.cuda.stream(stream):
            inp.copy_(src, non_blocking=True)  # also casts float32 -> float16 for the FP16 engine
            ctx.execute_async_v3(stream.cuda_stream)
        return out

    return run, stream


def bench(fn, src, stream=None):
    for _ in range(WARMUP):
        fn(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn(src)
    (stream.synchronize() if stream else torch.cuda.synchronize())
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms per batch


def main():
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT).eval().to(dev)
    engines = {"TRT FP32": load_engine("resnet18_fp32.engine"),
               "TRT FP16": load_engine("resnet18_fp16.engine")}

    rows = []
    for b in BATCHES:
        src = torch.randn(b, 3, 224, 224, device=dev)

        with torch.inference_mode():
            ref = model(src)
            ms = bench(lambda x: model(x), src)
        rows.append(("PyTorch FP32", b, ms, b / ms * 1e3, 0.0))

        for name, eng in engines.items():
            run, stream = trt_infer_fn(eng, b)
            out = run(src).clone()
            stream.synchronize()
            # correctness check against PyTorch: top-1 agreement and max abs logit diff
            agree = (out.argmax(1) == ref.argmax(1)).float().mean().item()
            ms = bench(run, src, stream)
            rows.append((name, b, ms, b / ms * 1e3, agree))

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'impl':14s}{'batch':>6s}{'ms/batch':>11s}{'img/s':>10s}{'top1 agree':>12s}")
    for r in rows:
        print(f"{r[0]:14s}{r[1]:6d}{r[2]:11.3f}{r[3]:10.1f}{r[4]:12.3f}")
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "batch", "ms_per_batch", "images_per_s", "top1_agreement_vs_pytorch"])
        w.writerows(rows)
    print("wrote results.csv")


if __name__ == "__main__":
    main()
