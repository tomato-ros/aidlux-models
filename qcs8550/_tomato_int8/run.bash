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

fun_print_line "《番茄ROS机器人》C++ 调用 NPU 演示"

./code/cpp/run.bash

fun_print_line "《番茄ROS机器人》Python 调用 NPU 演示"

./code/python/run.bash

