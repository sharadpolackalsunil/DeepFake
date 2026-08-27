import torch
from torch.utils.data import DataLoader
import webdataset as wds

def build_dataloader(cfg, split="train", num_workers=12):
    # Dummy implementation - in a real scenario this would build the actual webdataset pipeline

    # Path to shards
    # shard_pattern = f"data/cache/{cfg.dataset}/{split}-%06d.tar"

    def decode_and_format(sample):
        # We expect frames.npy, conf.npy, label.cls
        return {
            "frames": torch.tensor(sample["frames.npy"]).float() / 255.0, # Simple normalization
            "conf": torch.tensor(sample["conf.npy"]),
            "label": torch.tensor(sample["label.cls"])
        }

    dataset = (
        wds.WebDataset(f"data/cache/{cfg.dataset}/{split}-%06d.tar", resampled=True if split=="train" else False)
        .decode()
        .map(decode_and_format)
        .batched(cfg.batch_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
