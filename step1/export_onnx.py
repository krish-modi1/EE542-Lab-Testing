"""Export a pretrained ResNet-18 to ONNX twice: once in FP32 and once in FP16.

TensorRT builds "strongly typed" engines: every layer runs in the precision stored in the
ONNX file. So the FP16 engine needs its own FP16 ONNX file.
"""
import torch
import torchvision

model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()

for dtype, path in [(torch.float32, "resnet18_fp32.onnx"), (torch.float16, "resnet18_fp16.onnx")]:
    model = model.to(dtype)
    example = torch.randn(1, 3, 224, 224, device="cuda", dtype=dtype)
    torch.onnx.export(
        model, example, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # any batch size
        opset_version=18,
    )
    print("wrote", path)
