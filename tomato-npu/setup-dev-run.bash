#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

source $CUR_DIR/setup-dev-env-fun.bash

fun_hi

fun_run_cpp_example

fun_run_python_example
