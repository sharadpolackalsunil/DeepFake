#!/bin/bash
set -e

echo "Starting preprocessing..."
python -m smsdt.preprocess.build_shards \
  --dataset ffpp --workers 14 --chunk-stride 4 --out data/cache/ffpp/
echo "Preprocessing finished."
