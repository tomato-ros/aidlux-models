#!/bin/bash

CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

echo "工作目录:$CUR_DIR"

export QAIRT_VER="2.48.40.260702"

export QAIRT_HOME="$CUR_DIR/tool/qairt/${QAIRT_VER}"

export ARM_CROSS_TOOL_DIR="$CUR_DIR/tool/arm_cross_tool"

fun_hi(){
    echo "-----------------hello owrld----------------------"
}

fun_init_dir(){

    mkdir -p ${CUR_DIR}/tool \
             ${CUR_DIR}/code \
             ${CUR_DIR}/code/cpp \
             ${CUR_DIR}/code/python \
             ${CUR_DIR}/model

}

fun_install_dep(){

    sudo apt update

    sudo apt install -y wget unzip
}

fun_download_qairt(){

    echo "--------------------------------------------------------------------------------"

    QAIRT_ZIP="v${QAIRT_VER}.zip"
    
    QAIRT_LOCAL_DIR="${CUR_DIR}/tool"
    
    QAIRT_DOWN_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QAIRT_VER}/v${QAIRT_VER}.zip
"

    echo "----------> download qair ${QAIRT_VER}"

    if ! [ -f $QAIRT_LOCAL_DIR/$QAIRT_ZIP ]; then
        wget -O $QAIRT_LOCAL_DIR/$QAIRT_ZIP $QAIRT_DOWN_URL
    fi

    unzip -o $QAIRT_LOCAL_DIR/$QAIRT_ZIP -d $QAIRT_LOCAL_DIR

    rm -rf $QAIRT_LOCAL_DIR/$QAIRT_ZIP    
}

fun_download_arm_cross_tool(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> download arm cross tool"

    ARM_CROSS_TOOL_DIR="$CUR_DIR/tool/arm_cross_tool"

    git clone https://github.com/tomato-ros/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu $ARM_CROSS_TOOL_DIR

}

fun_create_venv(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> create python venv"

    export PYTHON_SRC_DIR="$CUR_DIR/code/python"

    export PYTHON_VENV_PATH="$PYTHON_SRC_DIR/.venv"

    python3.10 -m venv $PYTHON_VENV_PATH
}

fun_venv_install_dep(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> init python venv dep"

    # pip freeze > requirements.txt

    pip install -i https://mirrors.aliyun.com/pypi/simple -r $PYTHON_SRC_DIR/requirements.txt
}

fun_activate_venv(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> activate python venv"

    export PYTHON_SRC_DIR="$CUR_DIR/code/python"

    export PYTHON_VENV_PATH="$PYTHON_SRC_DIR/.venv"

    source $PYTHON_VENV_PATH/bin/activate

    python3 --version

}

fun_deactivate_venv(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> deactivate"

    deactivate
}

fun_view_onnx_model(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> view onnx model file"

    export ONNX_FILE="run_data/example/simple_net.onnx"
  
    netron $ONNX_FILE
}

fun_export_torch_onnx_model(){

  echo "--------------------------------------------------------------------------------"

  echo "----------> export torch onnx model"

  python3 code/python/src/generic_calib_data.py

  python3 code/python/src/torch_model_export_demo.py
}

fun_fix_qairt_env(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> fix qairt env"

    FIX_QAIRT_FILE="$QAIRT_HOME/bin/x86_64-linux-clang/qnn-model-lib-generator"
    
    # sys_root='${QNN_AARCH64_UBUNTU_GCC_94}')
    # sys_root='${QNN_AARCH64_UBUNTU_GCC_94}/aarch64-none-linux-gnu/libc')

    sed -i "s#sys_root='\${QNN_AARCH64_UBUNTU_GCC_94}')#sys_root='\${QNN_AARCH64_UBUNTU_GCC_94}/aarch64-none-linux-gnu/libc')#g" $FIX_QAIRT_FILE

}

fun_source_qairt_env(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> source qairt env"

    source $QAIRT_HOME/bin/envsetup.sh
    
    sudo $QAIRT_HOME/bin/check-linux-dependency.sh
    
    $QAIRT_HOME/bin/envcheck -c
    
    export PATH=$PATH:$QAIRT_SDK_ROOT/bin/x86_64-linux-clang

    $QAIRT_HOME/bin/envcheck -a
}

fun_qnn_onnx_converter_fp16(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> convert onnx to qnn fp16 cpp"

    export ONNX_FILE="run_data/example/simple_net.onnx"

    export ONNX_FILE_COV="run_data/example/simple_net_qnn_fp16"

    qnn-onnx-converter \
        --input_network $ONNX_FILE \
        -o $ONNX_FILE_COV \
        -d "input" "1,16" \
        --input_layout "input" NF \
        --validate_models
}


fun_qnn_onnx_converter_int8(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> convert onnx to qnn int8 cpp"

    export ONNX_FILE="run_data/example/simple_net.onnx"

    export ONNX_FILE_COV="run_data/example/simple_net_qnn_int8"
    
    export ONNX_CALIB_FILE="run_data/example/calib.txt"

    qnn-onnx-converter \
        --input_network $ONNX_FILE \
        -o "${ONNX_FILE_COV}" \
        --input_layout "input" NF \
        --input_list ${ONNX_CALIB_FILE} \
        --act_bitwidth 8 \
        --weights_bitwidth 8 \
        --bias_bitwidth 32 \
        --use_per_row_quantization

}


fun_qnn_build_int8(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> convert qnn build int8 so"

    export QNN_AARCH64_UBUNTU_GCC_94=$ARM_CROSS_TOOL_DIR
    export PATH=${QNN_AARCH64_UBUNTU_GCC_94}/bin:$PATH
    export MODEL_FILE="$CUR_DIR/run_data/example/simple_net_qnn_int8"
    export OUTPUT_LIB="$CUR_DIR/run_data/example/simple_net_qnn_int8_lib"

    cp -f $MODEL_FILE "${MODEL_FILE}.cpp"
    
    $QAIRT_HOME/bin/x86_64-linux-clang/qnn-model-lib-generator \
        -c "${MODEL_FILE}.cpp" \
        -b "${MODEL_FILE}.bin" \
        -t aarch64-ubuntu-gcc9.4 \
        -l libsimplenet_int8 \
        -o ${OUTPUT_LIB} \
        -d
    
    cp -r -f $OUTPUT_LIB/* $CUR_DIR/model
}

fun_qnn_build_fp16(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> convert qnn build fp16 so"

    export QNN_AARCH64_UBUNTU_GCC_94=$ARM_CROSS_TOOL_DIR
    export PATH=${QNN_AARCH64_UBUNTU_GCC_94}/bin:$PATH
    export MODEL_FILE="$CUR_DIR/run_data/example/simple_net_qnn_fp16"
    export OUTPUT_LIB="$CUR_DIR/run_data/example/simple_net_qnn_fp16_lib"

    cp -f $MODEL_FILE "${MODEL_FILE}.cpp"
    
    $QAIRT_HOME/bin/x86_64-linux-clang/qnn-model-lib-generator \
        -c "${MODEL_FILE}.cpp" \
        -b "${MODEL_FILE}.bin" \
        -t aarch64-ubuntu-gcc9.4 \
        -l libsimplenet_fp16 \
        -o ${OUTPUT_LIB} \
        -d
    
    cp -r -f $OUTPUT_LIB/* $CUR_DIR/model
}

fun_run_cpp_example(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> run cpp example (require qcs8550 board)"

    ./code/cpp/run.bash
}

fun_run_python_example(){

    echo "--------------------------------------------------------------------------------"

    echo "----------> run python example (require qcs8550 board)"

    python code/python/src/aidlite_demo.py model/aarch64-ubuntu-gcc9.4/libsimplenet_int8.so
    
    python code/python/src/aidlite_demo.py model/aarch64-ubuntu-gcc9.4/libsimplenet_fp16.so

}

