# Step 1: TensorRT CNN on a GPU buffer

Untested: written without GPU access. Expect small fixes.

## Run order
```bash
pip install torch torchvision onnx onnxscript tensorrt
bash setup_check.sh            # 0. confirm GPU, CUDA, TensorRT, PyTorch
python3 export_onnx.py         # 1a. resnet18.onnx (FP32) + resnet18_fp16.onnx
python3 build_engine.py && python3 build_engine.py fp16   # 1b. FP32 + FP16 engines
python3 gpu_buffer_runner.py   # 1c. benchmark + correctness -> results.csv
```
`build_engines.sh` (trtexec) is an alternative for 1b; it uses the `--fp16` flag and may not
match TensorRT 11 behaviour.

## Notes from first run (RTX 3060, TensorRT 11.3, torch 2.14)
- torch's ONNX exporter now needs `onnxscript`; opset is raised to 18 (warning is harmless).
- TensorRT 11 removed `BuilderFlag.FP16`. Engines are built "strongly typed": precision comes
  from the ONNX graph, so FP16 uses a separate FP16 ONNX export.

## Evidence to keep
- `logs/trtexec_fp32.log`, `logs/trtexec_fp16.log`
- `results.csv` (PyTorch vs TRT FP32 vs TRT FP16, batch 1/8/32, top-1 agreement)
- GPU name and driver from `setup_check.sh`

## Expected
- FP32 top-1 agreement with PyTorch should be ~1.0; FP16 slightly below 1.0 is normal.
- TRT should beat PyTorch, FP16 most on T4/L4 (tensor cores).
