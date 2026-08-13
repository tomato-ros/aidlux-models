#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

########################################################################################################################

rm -rf ./build

mkdir build && cd build

cmake .. && make -j$(nproc)

########################################################################################################################

#export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

./aidlite_demo ../../../models/QCS8550/FP16/yolov8m_qcs8550_fp16.qnn236.ctx.bin \
               ../data/bus.jpg \
               result.jpg

