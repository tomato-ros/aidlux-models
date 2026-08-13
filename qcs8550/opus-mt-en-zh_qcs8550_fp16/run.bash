#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

# echo "----------------------------------------"

# pip install -r code/requirements.txt

echo "----------------------------------------"

python code/main.py
