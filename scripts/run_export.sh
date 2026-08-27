#!/bin/bash
set -e

echo "Starting ONNX export..."
python -m smsdt.export.to_onnx --ckpt outputs/checkpoints/epoch0.pt --out outputs/smsdt.onnx

echo "Starting TensorRT engine build..."
python -m smsdt.export.build_trt_engine --onnx outputs/smsdt.onnx --out outputs/smsdt.engine
echo "Export pipeline finished."
