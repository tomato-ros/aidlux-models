#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

echo "********************************************************************"

export SOC=qcs8550

echo "********************************************************************"

fun_init(){

    echo "init..."

    mkdir -p $SOC

    # sudo aid-pkg update
    # sudo aid-pkg install aidlite-sdk
    # sudo aid-pkg install aidlite-qnn248
    # aid-pkg list
    
    # mms login
}

fun_down(){
    
    mms get -m opus-mt-en-zh -p FP16 -c $SOC -b QNN2.36 -d $SOC

    model_file=opus-mt-en-zh_qcs8550_fp16

    unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    rm -f $SOC/${model_file}.zip
}

echo "********************************************************************"

fun_init

fun_down

echo "********************************************************************"

echo "success"
