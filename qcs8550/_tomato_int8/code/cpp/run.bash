#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "----------------------------------------"

rm -rf ./build

mkdir build && cd build

cmake .. && make -j$(nproc)

./aidlite_demo ../../../models/QCS8550/int8/libsimplenet_int8.so

