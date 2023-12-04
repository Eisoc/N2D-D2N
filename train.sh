#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -i train.py  \
  --config "./config/train.json"
