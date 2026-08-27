import webdataset as wds
import numpy as np
import io

def write_shard(chunks, shard_path):
    with wds.TarWriter(shard_path) as sink:
        for i, c in enumerate(chunks):
            sink.write({
                "__key__": f"{c['video_id']}_{c['start_frame']:06d}",
                "frames.npy": c["frames"].astype(np.uint8),      # (T,224,224,3)
                "conf.npy": c["conf"].astype(np.float32),        # (T,)
                "label.cls": c["label"],                          # 0 real / 1 fake
                "manip.txt": c["manipulation_type"],
            })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ffpp")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--chunk-stride", type=int, default=4)
    parser.add_argument("--out", type=str, default="data/cache/ffpp/")
    args = parser.parse_args()
    print("Preprocessing shards with args:", args)
    # The actual implementation of shard building from raw videos goes here
