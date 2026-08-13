#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "----------------------------------------"

# sudo aid-pkg update

# sudo aid-pkg -i aidgen-sdk
# sudo aid-pkg -i aidgen-qnn240
# sudo aid-pkg -i aidgen-qnn248

# sudo aid-pkg -i aidlite-sdk
# sudo aid-pkg -i aidlite-qnn240
# sudo aid-pkg -i aidlite-qnn248

echo "----------------------------------------"

cd ./qnn248_qcs8550_cl4096/examples 

rm -rf build 

mkdir -p build && cd build

cmake .. && make -j$(nproc)

echo "----------------------------------------"

cp ./test_qwen3vl ../../

cd ../../

./test_qwen3vl qnn248

