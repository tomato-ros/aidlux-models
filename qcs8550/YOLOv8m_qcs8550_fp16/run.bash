#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "----------------------------------------"

python3 code/python/run_test.py \
    --target_model ./models/QCS8550/FP16/yolov8m_qcs8550_fp16.qnn236.ctx.bin \
    --imgs ./code/python/bus.jpg \
    --invoke_nums 10
