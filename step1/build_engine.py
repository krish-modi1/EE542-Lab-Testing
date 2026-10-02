"""Build a TensorRT engine from resnet18_<precision>.onnx.

Usage: python3 build_engine.py fp32
       python3 build_engine.py fp16
"""
import sys

import tensorrt as trt

precision = sys.argv[1]  # "fp32" or "fp16"
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# Strongly typed: each layer keeps the data type from the ONNX file.
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser = trt.OnnxParser(network, logger)
if not parser.parse(open(f"resnet18_{precision}.onnx", "rb").read()):
    sys.exit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

# The engine accepts batches of 1 to 32 images and is tuned for 8.
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
config = builder.create_builder_config()
config.add_optimization_profile(profile)

engine = builder.build_serialized_network(network, config)
open(f"resnet18_{precision}.engine", "wb").write(engine)
print(f"wrote resnet18_{precision}.engine")
