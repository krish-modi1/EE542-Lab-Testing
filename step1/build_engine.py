"""Fallback for Step 1b when trtexec is not installed (pip `tensorrt` has no trtexec).
Usage: python3 build_engine.py [fp16]
Written against the TensorRT 10 Python API.
"""
import sys
import tensorrt as trt

fp16 = len(sys.argv) > 1 and sys.argv[1] == "fp16"
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)  # explicit batch is the default in TRT 10
parser = trt.OnnxParser(network, logger)

with open("resnet18.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
if fp16:
    config.set_flag(trt.BuilderFlag.FP16)

profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
config.add_optimization_profile(profile)

engine_bytes = builder.build_serialized_network(network, config)
out = f"resnet18_{'fp16' if fp16 else 'fp32'}.engine"
with open(out, "wb") as f:
    f.write(engine_bytes)
print("wrote", out)
