#!/bin/bash
set -e

echo "Starting training..."
# export PYTHONPATH=.
python -m smsdt.train
echo "Training finished."
