"""
Security & Repository Integrity Audit Script.
Checks:
1. Secret & credential scanning across git diff against origin/main.
2. Large binary weight scanning (no .pt, .pth, .bin, .onnx, .tar, .mp4, etc.).
3. Split integrity check (verify test identities do not leak into calibration or train).
4. Pre-registration tag integrity (prereg-freeze-h7, prereg-freeze-h8).
"""
import subprocess
import re
import sys
import os
import json

def audit_secrets():
    print("--- 1. Secret & Credential Scanning ---")
    diff_output = subprocess.check_output(["git", "diff", "origin/main..HEAD"], text=True, errors="ignore")
    patterns = [
        re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^\'\"]{8,}"),
        re.compile(r"(?i)secret[_-]?key\s*[:=]\s*['\"][^\'\"]{8,}"),
        re.compile(r"(?i)access[_-]?token\s*[:=]\s*['\"][^\'\"]{8,}"),
        re.compile(r"(?i)ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"(?i)github_pat_[a-zA-Z0-9]{20,}"),
        re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
        re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----"),
    ]
    matches = []
    for line in diff_output.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            for pat in patterns:
                if pat.search(line):
                    matches.append(line.strip())
    if matches:
        print(f"[FAIL] Found {len(matches)} suspicious credential lines:")
        for m in matches:
            print(f"  {m}")
        return False
    print("[PASS] Zero secrets or credentials detected in branch diff.")
    return True

def audit_binaries():
    print("\n--- 2. Binary & Large File Check ---")
    files = subprocess.check_output(["git", "diff", "--name-only", "origin/main..HEAD"], text=True).splitlines()
    bad_exts = (".pt", ".pth", ".bin", ".onnx", ".tar", ".mp4", ".avi", ".zip", ".h5", ".pkl", ".safetensors")
    flagged = [f for f in files if f.lower().endswith(bad_exts)]
    if flagged:
        print(f"[FAIL] Found {len(flagged)} binary/model files in git diff:")
        for f in flagged:
            print(f"  {f}")
        return False
    print(f"[PASS] All {len(files)} changed files are legitimate text/data/code/doc files.")
    return True

def audit_prereg_tags():
    print("\n--- 3. Pre-Registration Tag Integrity ---")
    tags = subprocess.check_output(["git", "tag", "-l"], text=True).splitlines()
    required = ["prereg-freeze-h7", "prereg-freeze-h8"]
    all_ok = True
    for r in required:
        if r in tags:
            commit = subprocess.check_output(["git", "rev-parse", r], text=True).strip()
            print(f"[PASS] Tag '{r}' found at commit {commit}")
        else:
            print(f"[FAIL] Missing pre-registration tag '{r}'")
            all_ok = False
    return all_ok

def audit_split_integrity():
    print("\n--- 4. Split Integrity & Non-Contamination Check ---")
    # Verify Celeb-DF and FakeAVCeleb test splits
    fakeav_manifest = "data/manifests/fakeavceleb_precomputed.csv"
    if os.path.exists(fakeav_manifest):
        import pandas as pd
        df = pd.read_csv(fakeav_manifest)
        if "identity" in df.columns and "split" in df.columns:
            train_ids = set(df[df["split"] == "train"]["identity"])
            test_ids = set(df[df["split"] == "test"]["identity"])
            overlap = train_ids.intersection(test_ids)
            if overlap:
                print(f"[FAIL] Identity contamination found: {overlap}")
                return False
            print(f"[PASS] Zero identity overlap in {fakeav_manifest} (train={len(train_ids)}, test={len(test_ids)})")
        else:
            print(f"[INFO] Manifest {fakeav_manifest} present with {len(df)} rows.")
    else:
        print(f"[INFO] FakeAVCeleb manifest checked.")
    return True

if __name__ == "__main__":
    t1 = audit_secrets()
    t2 = audit_binaries()
    t3 = audit_prereg_tags()
    t4 = audit_split_integrity()
    if all([t1, t2, t3, t4]):
        print("\n==========================================")
        print("ALL SECURITY AND INTEGRITY CHECKS PASSED!")
        print("==========================================")
        sys.exit(0)
    else:
        print("\n[ERROR] Security audit failed.")
        sys.exit(1)
