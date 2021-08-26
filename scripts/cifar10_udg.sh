#!/bin/bash
DATA_DIR=$1
OUTPUT_DIR=$2

python train.py \
    --config configs/train/cifar10_udg.yml \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}

python test.py \
    --config configs/test/cifar10.yml \
    --checkpoint ${OUTPUT_DIR}/best.ckpt \
    --data_dir ${DATA_DIR} \
    --csv_path ${OUTPUT_DIR}/results.csv