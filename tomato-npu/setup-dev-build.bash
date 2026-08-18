#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

source $CUR_DIR/setup-dev-env-fun.bash

fun_hi

fun_activate_venv

fun_export_torch_onnx_model

# fun_view_onnx_model

fun_source_qairt_env

fun_qnn_onnx_converter_fp16

fun_qnn_onnx_converter_int8

fun_qnn_build_int8

fun_qnn_build_fp16

fun_deactivate_venv
