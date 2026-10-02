"""Run the TensorRT engines on input that is already in GPU memory, and compare their
speed and predictions with PyTorch. Writes results.csv.

Why GPU memory: with DOCA GPUNetIO the network card writes packets straight into GPU
memory, so the inference code must take a GPU address, not a CPU array.

Step 2 imports load_engine() and make_runner() from this file.
"""
import csv
import time

import tensorrt as trt
import torch
import torchvision

TORCH_DTYPE = {trt.float32: torch.float32, trt.float16: torch.float16}


def load_engine(path):
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(open(path, "rb").read())


def make_runner(engine, batch):
    """Allocate GPU input/output buffers for `batch` images. Returns run(x) -> logits."""
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", (batch, 3, 224, 224))
    inp = torch.empty((batch, 3, 224, 224), device="cuda",
                      dtype=TORCH_DTYPE[engine.get_tensor_dtype("input")])
    out = torch.empty((batch, 1000), device="cuda",
                      dtype=TORCH_DTYPE[engine.get_tensor_dtype("logits")])
    ctx.set_tensor_address("input", inp.data_ptr())  # TensorRT reads this GPU address
    ctx.set_tensor_address("logits", out.data_ptr())
    stream = torch.cuda.Stream()

    def run(x):
        stream.wait_stream(torch.cuda.current_stream())  # x must be ready before we copy it
        with torch.cuda.stream(stream):
            inp.copy_(x)  # with GPUNetIO, the network card would fill `inp` instead
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()  # wait for TensorRT before anyone reads `out`
        return out.float()

    return run


def time_ms(fn, x, iters=200, warmup=20):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


if __name__ == "__main__":
    model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
    engines = {"TRT FP32": load_engine("resnet18_fp32.engine"),
               "TRT FP16": load_engine("resnet18_fp16.engine")}

    rows = []
    for batch in (1, 8, 32):
        # Random input is enough: we measure speed and whether TensorRT agrees with PyTorch.
        x = torch.randn(batch, 3, 224, 224, device="cuda")
        with torch.inference_mode():
            ref = model(x)
            rows.append(["PyTorch FP32", batch, time_ms(model, x), "ref", "ref"])
        for name, engine in engines.items():
            run = make_runner(engine, batch)
            y = run(x)
            agree = (y.argmax(1) == ref.argmax(1)).float().mean().item()
            diff = (y - ref).abs().max().item()
            rows.append([name, batch, time_ms(run, x), f"{agree:.3f}", f"{diff:.4f}"])

    print("GPU:", torch.cuda.get_device_name())
    print(f"{'impl':14}{'batch':>6}{'ms':>9}{'img/s':>9}{'top1':>8}{'maxdiff':>9}")
    for impl, batch, ms, agree, diff in rows:
        print(f"{impl:14}{batch:>6}{ms:>9.3f}{batch / ms * 1000:>9.0f}{agree:>8}{diff:>9}")

    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["impl", "batch", "ms_per_batch", "top1_agreement", "max_logit_diff"])
        writer.writerows([[i, b, f"{ms:.3f}", a, d] for i, b, ms, a, d in rows])
    print("wrote results.csv")
