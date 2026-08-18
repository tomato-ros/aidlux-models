#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "----------------------------------------"

rm -rf ./build

mkdir build && cd build

cmake .. && make -j$(nproc)

./aidlite_demo ../../../model/aarch64-ubuntu-gcc9.4/libsimplenet_int8.so

./aidlite_demo ../../../model/aarch64-ubuntu-gcc9.4/libsimplenet_fp16.so
