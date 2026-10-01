#!/usr/bin/env bash
# Step 0: confirm the instance has what Step 1 needs.
set -u

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "MISSING: nvidia-smi"

echo "== CUDA compiler =="
nvcc --version 2>/dev/null | tail -n 1 || echo "nvcc not found (fine for Step 1, needed for Step 2 GPUNetIO build)"

echo "== trtexec =="
if command -v trtexec >/dev/null; then trtexec --help 2>&1 | head -n 1
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then echo "found at /usr/src/tensorrt/bin/trtexec (add to PATH)"
else echo "trtexec not found -> use build_engine.py instead"; fi

echo "== Python packages =="
python3 - <<'EOF'
for m in ("torch", "torchvision", "onnx", "tensorrt"):
    try:
        mod = __import__(m)
        print(f"{m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({e.__class__.__name__})")
try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception:
    pass
EOF

echo
echo "If anything is missing:  pip install torch torchvision onnx tensorrt"
