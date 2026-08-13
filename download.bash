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

fun_down_opus_mt_en_zh(){
    
    mms get -m opus-mt-en-zh -p FP16 -c $SOC -b QNN2.36 -d $SOC

    model_file=opus-mt-en-zh_qcs8550_fp16

    unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    rm -f $SOC/${model_file}.zip
}

fun_down_YOLOv8m(){
    
    mms get -m YOLOv8m -p FP16 -c $SOC -b QNN2.36 -d $SOC

    model_file=YOLOv8m_qcs8550_fp16

    unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    rm -f $SOC/${model_file}.zip
}

# 预览版加密模型
fun_down_MeloTTS_Chinese(){
    
    mms get -m MeloTTS-Chinese -p FP16 -c $SOC -b QNN2.31 -d $SOC

    model_file=MeloTTS-Chinese_qcs8550_fp16

    unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    rm -f $SOC/${model_file}.zip
}

fun_down_WeTTS(){
    
    mms get -m WeTTS -p FP32 -c $SOC -b ONNX -d $SOC

    model_file=WeTTS_qcs8550_fp32

    unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    rm -f $SOC/${model_file}.zip
}

fun_down_Qwen3_VL_8B_Instruct(){
    
    mms get -m Qwen3-VL-8B-Instruct -p W4A16 -c $SOC -b QNN2.48 -d $SOC

    # model_file=Qwen3-VL-8B-Instruct

    # unzip -o $SOC/${model_file}.zip -d $SOC/${model_file}

    # rm -f $SOC/${model_file}.zip
}



echo "********************************************************************"

fun_init

# fun_down_opus_mt_en_zh

# fun_down_YOLOv8m

# fun_down_MeloTTS_Chinese

# fun_down_WeTTS

fun_down_Qwen3_VL_8B_Instruct

echo "********************************************************************"

echo "success"
