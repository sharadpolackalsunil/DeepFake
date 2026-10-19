"""Fail loudly if any face identity appears in more than one split.

Usage: python tools/check_split.py data/splits/index.csv
index.csv columns: video_id, split, label, method, target_id, source_id
(source_id is empty for real videos and for reenactment methods).
"""
import sys
import csv
from collections import defaultdict


def identities(row):
    ids = {row["target_id"]}
    if row.get("source_id"):
        ids.add(row["source_id"])
    return ids


def main(path="data/splits/index.csv"):
    owner = defaultdict(set)          # identity -> set of splits it appears in
    counts = defaultdict(lambda: defaultdict(int))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["split"]][row["method"]] += 1
            for ident in identities(row):
                owner[ident].add(row["split"])
    leaks = {i: s for i, s in owner.items() if len(s) > 1}
    for split, per_method in sorted(counts.items()):
        print(f"Split [{split}]:", dict(per_method))
    if leaks:
        print(f"\n[LEAK DETECTED]: {len(leaks)} identities appear in >1 split!")
        for ident, splits in list(leaks.items())[:5]:
            print(f"  - Identity {ident}: present in {splits}")
        sys.exit(1)
    print("\n[OK]: Splits are strictly identity-disjoint across all methods!")


if __name__ == "__main__":
    split_path = sys.argv[1] if len(sys.argv) > 1 else "data/splits/index.csv"
    main(split_path)
