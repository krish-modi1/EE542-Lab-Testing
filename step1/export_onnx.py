"""Step 1a: export a pretrained ResNet-18 to ONNX with a dynamic batch dimension."""
import torch
import torchvision

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print("wrote resnet18.onnx")
