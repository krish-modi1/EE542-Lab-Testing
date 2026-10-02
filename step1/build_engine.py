"""Step 1b: build a TensorRT engine from ONNX (use when trtexec is not installed).
Usage: python3 build_engine.py [fp16]

TensorRT 11: BuilderFlag.FP16 is gone. The network is built "strongly typed", so each layer
runs in the precision of the ONNX graph: resnet18.onnx -> FP32 engine,
resnet18_fp16.onnx -> FP16 engine. Falls back to a plain network on older TensorRT.
"""
import sys

import tensorrt as trt

fp16 = len(sys.argv) > 1 and sys.argv[1] == "fp16"
onnx_path = "resnet18_fp16.onnx" if fp16 else "resnet18.onnx"
out = f"resnet18_{'fp16' if fp16 else 'fp32'}.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

flags = 0
if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
config.add_optimization_profile(profile)

engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    sys.exit("engine build failed")
with open(out, "wb") as f:
    f.write(engine_bytes)
print(f"wrote {out} (from {onnx_path}, strongly_typed={bool(flags)})")
