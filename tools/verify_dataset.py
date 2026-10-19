"""Verify FaceForensics++ c23 raw dataset integrity and directory layout.

Usage:
    python tools/verify_dataset.py --root data/raw/ffpp_c23
"""
import os
import argparse
import glob

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
OPTIONAL_METHODS = ["DeepFakeDetection"]

def check_video_dir(path, name, expected_min=140):
    if not os.path.exists(path):
        return False, 0, f"Directory does not exist: {path}"
    
    mp4s = glob.glob(os.path.join(path, "*.mp4"))
    count = len(mp4s)
    
    # Check non-empty files
    empty_files = [f for f in mp4s if os.path.getsize(f) == 0]
    
    status = f"Found {count} .mp4 videos"
    if empty_files:
        status += f" (WARNING: {len(empty_files)} 0-byte corrupt files!)"
    
    is_ok = count >= expected_min and len(empty_files) == 0
    return is_ok, count, status

def main():
    parser = argparse.ArgumentParser(description="Verify FF++ c23 raw dataset layout")
    parser.add_argument("--root", default="data/raw/ffpp_c23", help="Root directory of raw FF++ c23 dataset")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print("=" * 60)
    print(f"Verifying FF++ c23 Dataset at: {root}")
    print("=" * 60)

    if not os.path.exists(root):
        print(f"\n[ERROR] Root path does not exist: {root}")
        print("Please verify the drive letter or folder location.")
        return

    # 1. Real videos (original_sequences/youtube/c23/videos)
    real_path = os.path.join(root, "original_sequences", "youtube", "c23", "videos")
    ok, count, msg = check_video_dir(real_path, "Real (youtube)", expected_min=1000)
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} Pristine Reals (youtube): {msg} (Target: 1,000 videos)")

    # 2. Manipulated methods
    for m in METHODS:
        fake_path = os.path.join(root, "manipulated_sequences", m, "c23", "videos")
        ok, count, msg = check_video_dir(fake_path, m, expected_min=1000)
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} Manipulation '{m}': {msg} (Target: 1,000 videos)")

    # 3. Optional DFD (quarantined cross-dataset benchmark)
    dfd_path = os.path.join(root, "manipulated_sequences", "DeepFakeDetection", "c23", "videos")
    ok, count, msg = check_video_dir(dfd_path, "DeepFakeDetection", expected_min=100)
    tag = "[INFO]" if ok else "[NOT FOUND]"
    print(f"{tag} Cross-Dataset 'DeepFakeDetection': {msg} (Excluded from FF++ splits)")

    actors_path = os.path.join(root, "original_sequences", "actors", "c23", "videos")
    ok, count, msg = check_video_dir(actors_path, "Actors Reals", expected_min=100)
    tag = "[INFO]" if ok else "[NOT FOUND]"
    print(f"{tag} Cross-Dataset 'Actors Reals': {msg} (Excluded from FF++ splits)")
    print("=" * 60)

if __name__ == "__main__":
    main()
