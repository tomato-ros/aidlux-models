#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "----------------------------------------"

# pip install onnxruntime===1.18.0

echo "----------------------------------------"

python ./code/python/run_test.py \
    --text_path "窗前明月光，疑是地上霜，举头望明月，低头思故乡" \
    --wav_path ./test_data_result/results_1

python ./code/python/run_test.py \
    --text_path ./code/python/chinese.txt \
    --wav_path ./test_data_result/results_2

