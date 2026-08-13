#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

#/////////////////////////////////////////////////////////////////////////////////////////////////

fun_print_line(){
    echo -e "\033[1;31m================================================================================\033[0m"
}

#/////////////////////////////////////////////////////////////////////////////////////////////////

fun_print_line

export SOC=qcs8550

fun_print_line

# 英文翻译
./$SOC/opus-mt-en-zh_qcs8550_fp16/run.bash

fun_print_line

# yolov8m
./$SOC/YOLOv8m_qcs8550_fp16/run.bash

fun_print_line






