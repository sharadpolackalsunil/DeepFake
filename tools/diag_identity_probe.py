"""Diagnostic D3: Identity Encoding Probe (Dong et al., CVPR 2023).

Trains a linear probe on frozen pooled features of real videos to predict face identity
(720 classes). A high top-1 accuracy indicates the model memorizes facial identity rather
than learning invariant forensic artifact representations.
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader

from smsdt.eval_cross import load_model
from smsdt.data.video_dataset import VideoStore, contiguous_starts, load_index


def extract_features(model, model_type, rows, store_root, device, norm_fn):
    features, labels, id_map = [], [], {}
    curr_id = 0

    print(f"Extracting features from {len(rows)} real videos...")
    with torch.no_grad():
        for r in rows:
            ident = r["target_id"]
            if ident not in id_map:
                id_map[ident] = curr_id
                curr_id += 1

            vs = VideoStore(store_root, r["video_id"])
            starts = contiguous_starts(vs.fidx, T=8, stride=1)
            if not starts:
                continue

            s = starts[len(starts) // 2]
            frames_u8 = np.asarray(vs.frames[s:s + 8])
            H, W = frames_u8.shape[1:3]
            cy, cx = (H - 224) // 2, (W - 224) // 2
            cropped = frames_u8[:, cy:cy + 224, cx:cx + 224]
            tensor = torch.from_numpy(np.ascontiguousarray(cropped)).permute(0, 3, 1, 2).float() / 255.0
            if norm_fn is not None:
                tensor = norm_fn(tensor)

            tensor = tensor.unsqueeze(0).to(device)  # (1, 8, 3, 224, 224)

            # Extract pooled representation
            if model_type == "v2":
                cls, _ = model.encoder(tensor.view(8, 3, 224, 224))
                feat = cls.mean(dim=0).cpu().numpy()
            else:
                # v1: extract from spatial/temporal representation
                tokens, _ = model._extract_tokens(tensor)
                feat = tokens.mean(dim=(1, 2)).squeeze(0).cpu().numpy()

            features.append(feat)
            labels.append(id_map[ident])

    return np.array(features), np.array(labels), id_map


def main():
    parser = argparse.ArgumentParser(description="Linear identity probe on real face representations")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--index", type=str, default="data/splits/index.csv")
    parser.add_argument("--store", type=str, default="data/store")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type, norm_fn = load_model(args.ckpt, device)

    train_rows = [r for r in load_index(args.index, "train") if r["method"] == "real"]
    val_rows = [r for r in load_index(args.index, "val") if r["method"] == "real"]

    X_train, y_train, id_map = extract_features(model, model_type, train_rows, args.store, device, norm_fn)
    print(f"Extracted X_train: {X_train.shape}, Unique Identities: {len(id_map)}")

    # Fit linear probe
    print("Fitting linear identity probe...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    chance_level = 1.0 / len(id_map)
    print(f"\nIdentity Probe Train Accuracy: {train_acc * 100:.2f}% (Chance level: {chance_level * 100:.2f}%)")

    if train_acc > 0.50:
        print("[WARNING] High identity probe accuracy (>50%) indicates the backbone strongly encodes facial identity!")
    else:
        print("[SUCCESS] Low/moderate identity encoding: features are largely identity-agnostic.")


if __name__ == "__main__":
    main()
