#!/bin/bash

rm -r outputs
mkdir outputs
make clean
make
srun -A ACD114118 --gpus-per-node=1 ./final_gpu
srun -A ACD114118 --gpus-per-node=1 python3 validation.py
