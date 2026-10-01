# Step 1: TensorRT CNN on a GPU buffer

Untested: written without GPU access. Expect small fixes.

## Run order
```bash
bash setup_check.sh            # 0. confirm GPU, CUDA, TensorRT, PyTorch
python3 export_onnx.py         # 1a. resnet18.onnx
bash build_engines.sh          # 1b. FP32 + FP16 engines, logs/ (needs trtexec)
#   no trtexec?  python3 build_engine.py && python3 build_engine.py fp16
python3 gpu_buffer_runner.py   # 1c. benchmark + correctness -> results.csv
```

## Evidence to keep
- `logs/trtexec_fp32.log`, `logs/trtexec_fp16.log`
- `results.csv` (PyTorch vs TRT FP32 vs TRT FP16, batch 1/8/32, top-1 agreement)
- GPU name and driver from `setup_check.sh`

## Expected
- FP32 top-1 agreement with PyTorch should be ~1.0; FP16 slightly below 1.0 is normal.
- TRT should beat PyTorch, FP16 most on T4/L4 (tensor cores).
