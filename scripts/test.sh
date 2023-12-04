#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -i debug.py  \
  --batchsize 4 \
  --n-ref 3 \
  --upsample-flow-output \
  --lr 5e-4 \
  --max-depth 5e3 \
  --depth-mask-weight 0.1 --flow-mask-weight 0.1 \
  --verbose  \
  --load-path
