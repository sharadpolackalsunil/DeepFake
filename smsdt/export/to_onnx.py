import torch
import os
from smsdt.models.smsdt import SMSDT

def export_to_onnx(ckpt_path, out_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Exporting to ONNX on {device}...")
    model = SMSDT().eval().to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found. Exporting random weights.")

    dummy_frames = torch.randn(1, 8, 3, 224, 224, device=device)
    dummy_conf = torch.ones(1, 8, device=device)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.onnx.export(
        model, (dummy_frames, dummy_conf), out_path,
        input_names=["frames", "conf"], output_names=["logit"],
        opset_version=18, dynamic_axes={"frames": {0: "batch"}, "conf": {0: "batch"}},
    )
    print(f"Successfully exported to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--out", type=str, default="outputs/smsdt.onnx")
    args = parser.parse_args()
    export_to_onnx(args.ckpt, args.out)
