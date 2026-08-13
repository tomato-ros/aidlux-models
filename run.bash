#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

#/////////////////////////////////////////////////////////////////////////////////////////////////

fun_print_line(){
    local text="$1"
    local color="${2:-1;31}"
    local line="================================================================================"
    local width=${#line}
    local pad=$(( (width - ${#text}) / 2 - 1 ))
    local space=$(printf "%${pad}s")

    echo -e "\033[${color}m${line}\033[0m"
    echo -e "\033[${color}m|${space}${text}${space}|\033[0m"
    echo -e "\033[${color}m${line}\033[0m"
}

#/////////////////////////////////////////////////////////////////////////////////////////////////

export SOC=qcs8550

fun_print_line "opus-mt-en-zh_qcs8550_fp16"

./$SOC/opus-mt-en-zh_qcs8550_fp16/run.bash


fun_print_line "YOLOv8m_qcs8550_fp16"

./$SOC/YOLOv8m_qcs8550_fp16/run.bash

