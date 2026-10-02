"""Step 1a: export a pretrained ResNet-18 to ONNX (dynamic batch), in FP32 and FP16.

TensorRT 11 removed BuilderFlag.FP16: precision now comes from the ONNX graph's own types
("strongly typed" networks), so FP16 needs its own FP16 ONNX file.
"""
import copy

import torch
import torchvision

dev = "cuda" if torch.cuda.is_available() else "cpu"
base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()


def export(model, dtype, path):
    model = copy.deepcopy(model).to(dev, dtype)
    dummy = torch.randn(1, 3, 224, 224, device=dev, dtype=dtype)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print("wrote", path)


export(base, torch.float32, "resnet18.onnx")
export(base, torch.float16, "resnet18_fp16.onnx")
