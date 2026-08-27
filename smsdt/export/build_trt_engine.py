import os
import subprocess

def build_engine(onnx_path, out_engine_path):
    print(f"Building TensorRT engine from {onnx_path} to {out_engine_path}...")
    # NOTE: trtexec needs to be in PATH
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={out_engine_path}",
        "--fp16", # Fallback to fp16 for generic building script if fp8 is not supported locally
        "--shapes=frames:1x8x3x224x224,conf:1x8",
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Engine built successfully.")
    except Exception as e:
        print(f"Error building engine: {e}")
        print("Note: Ensure 'trtexec' is installed and in your PATH.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="outputs/smsdt.onnx")
    parser.add_argument("--out", type=str, default="outputs/smsdt_fp8.engine")
    args = parser.parse_args()
    build_engine(args.onnx, args.out)
