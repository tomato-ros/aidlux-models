#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

source $CUR_DIR/setup-dev-env-fun.bash

fun_hi

fun_init_dir

fun_install_dep

fun_download_qairt

fun_source_qairt_env

fun_fix_qairt_env

fun_download_arm_cross_tool

fun_create_venv

fun_activate_venv

fun_venv_install_dep

fun_deactivate_venv
