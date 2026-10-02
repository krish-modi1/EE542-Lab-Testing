#!/usr/bin/env bash
# Print the GPU, driver, CUDA compiler, and Python package versions this lab depends on.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvcc --version | tail -n 1
python3 -c "
import torch, torchvision, onnx, tensorrt
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('torchvision', torchvision.__version__, '| onnx', onnx.__version__, '| tensorrt', tensorrt.__version__)
"
