#!/usr/bin/env bash
# Step 1b: build FP32 and FP16 TensorRT engines with trtexec and save the logs as evidence.
# Shapes cover batch 1..32, optimised for 8.
set -euo pipefail
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
SHAPES="--minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:32x3x224x224"

mkdir -p logs
$TRTEXEC --onnx=resnet18.onnx $SHAPES --saveEngine=resnet18_fp32.engine        2>&1 | tee logs/trtexec_fp32.log
$TRTEXEC --onnx=resnet18.onnx $SHAPES --saveEngine=resnet18_fp16.engine --fp16 2>&1 | tee logs/trtexec_fp16.log

echo
echo "Throughput / latency summary:"
grep -hE "Throughput|Latency: min" logs/trtexec_fp32.log logs/trtexec_fp16.log
