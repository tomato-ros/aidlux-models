import time
import numpy as np
import aidlite
import warnings

# python main.py ../../../model/aarch64-ubuntu-gcc9.4/libsimplenet_int8.so
# python main.py ../../../model/aarch64-ubuntu-gcc9.4/libsimplenet_fp16.so

# 屏蔽警告，与SDK封装层风格对齐
warnings.filterwarnings("ignore", category=FutureWarning, module="importlib._bootstrap")

# ==================== 【配置常量区，统一维护】====================
# 命令行传入覆盖此路径
MODEL_PATH = "models/QCS8550/int8/libsimplenet_int8.so"

# SimpleNet 模型参数（与C++完全对齐）
BATCH_SIZE = 1
INPUT_DIM = 16
OUTPUT_DIM = 4

# Tensor索引
INPUT_TENSOR_IDX = 0
OUTPUT_TENSOR_IDX = 0

# INT8量化模型开关
IS_QUANT_MODEL = True
# is_native=False：用户传float，SDK内部自动量化/反量化（推荐）
IS_NATIVE_DATA = False

# 性能测试配置
WARMUP_COUNT = 10  # 预热次数
TEST_LOOP_COUNT = 50000  # 正式循环次数


# =================================================================


def aid_sdk_demo(model_path: str):
    ver = aidlite.get_library_version()
    print(f"Aidlite library version : {ver}")
    aidlite.set_log_level(aidlite.LogLevel.WARNING)

    # 1. 创建Model
    model = aidlite.Model.create_instance(model_path)
    if model is None:
        print("[ERROR] Create model instance failed!")
        return 1

    input_shapes = [[BATCH_SIZE, INPUT_DIM]]
    output_shapes = [[BATCH_SIZE, OUTPUT_DIM]]

    tensor_dtype = aidlite.DataType.TYPE_INT8 if IS_QUANT_MODEL else aidlite.DataType.TYPE_FLOAT32
    model.set_model_properties(
        input_shapes,
        tensor_dtype,
        output_shapes,
        tensor_dtype
    )

    # 2. 创建Config
    config = aidlite.Config.create_instance()
    if config is None:
        print("[ERROR] Create config failed!")
        return 1

    config.framework_type = aidlite.FrameworkType.TYPE_QNN
    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.is_quantify_model = 1 if IS_QUANT_MODEL else 0

    # 3. 构建解释器
    fast_interpreter = aidlite.InterpreterBuilder.build_interpreter_from_model_and_config(model, config)
    # model、config不再需要
    del model
    del config

    if fast_interpreter is None:
        print("[ERROR] Build interpreter failed!")
        return 1

    # 4. 初始化 & 加载模型
    ret = fast_interpreter.init()
    if ret != 0:
        print(f"[ERROR] interpreter init failed, ret={ret}")
        fast_interpreter.destroy()
        return 1

    ret = fast_interpreter.load_model()
    if ret != 0:
        print(f"[ERROR] load_model failed, ret={ret}")
        fast_interpreter.destroy()
        return 1

    # ======================== 推理准备 ========================
    input_elem_cnt = BATCH_SIZE * INPUT_DIM
    # 构造输入数组，C连续float32
    input_fp32 = np.full(shape=(BATCH_SIZE, INPUT_DIM), fill_value=0.1, dtype=np.float32)

    invoke_cost_ms = []

    # ========== 预热阶段 ==========
    print(f"\n===== Warmup stage, iterations: {WARMUP_COUNT} =====")
    for i in range(WARMUP_COUNT):
        ret = fast_interpreter.set_input_tensor(INPUT_TENSOR_IDX, input_fp32, is_native=IS_NATIVE_DATA)
        if ret != 0:
            print(f"[ERROR] set_input_tensor failed, ret={ret}")
            fast_interpreter.destroy()
            return 1

        ret = fast_interpreter.invoke()
        if ret != 0:
            print(f"[ERROR] invoke failed at warmup, ret={ret}")
            fast_interpreter.destroy()
            return 1

        # get_output_tensor 需要传入 output_type 参数
        out_arr = fast_interpreter.get_output_tensor(
            OUTPUT_TENSOR_IDX,
            aidlite.DataType.TYPE_FLOAT32,
            is_native=IS_NATIVE_DATA
        )
    print("Warmup finished!\n")

    # ========== 正式循环压测 ==========
    print(f"===== Start performance test, total loop: {TEST_LOOP_COUNT} =====")
    for iter_idx in range(TEST_LOOP_COUNT):
        ret = fast_interpreter.set_input_tensor(INPUT_TENSOR_IDX, input_fp32, is_native=IS_NATIVE_DATA)
        if ret != 0:
            print(f"[ERROR] set_input_tensor failed, ret={ret}")
            fast_interpreter.destroy()
            return 1

        # 只统计invoke推理耗时
        t_start = time.perf_counter()
        ret = fast_interpreter.invoke()
        t_end = time.perf_counter()

        if ret != 0:
            print(f"[ERROR] invoke failed, iter={iter_idx} ret={ret}")
            fast_interpreter.destroy()
            return 1

        cost_ms = (t_end - t_start) * 1000.0
        invoke_cost_ms.append(cost_ms)

        out_arr = fast_interpreter.get_output_tensor(
            OUTPUT_TENSOR_IDX,
            aidlite.DataType.TYPE_FLOAT32,
            is_native=IS_NATIVE_DATA
        )

        # 可选：每隔若干轮打印输出，取消注释启用
        # if (iter_idx + 1) % 50 == 0:
        #     print(f"[{iter_idx+1}/{TEST_LOOP_COUNT}] invoke cost: {cost_ms:.3f} ms | Output: {out_arr.flatten()}")

    # ========== 统计结果 ==========
    sum_ms = sum(invoke_cost_ms)
    avg_ms = sum_ms / TEST_LOOP_COUNT
    min_ms = min(invoke_cost_ms)
    max_ms = max(invoke_cost_ms)
    fps = 1000.0 / avg_ms

    print("\n===== Test Result Summary =====")
    print(f"Loop count      : {TEST_LOOP_COUNT}")
    print(f"Average invoke  : {avg_ms:.3f} ms")
    print(f"Min invoke time : {min_ms:.3f} ms")
    print(f"Max invoke time : {max_ms:.3f} ms")
    print(f"FPS estimate    : {fps:.2f}")

    # 释放资源
    ret = fast_interpreter.destroy()
    if ret != 0:
        print(f"[WARN] interpreter destroy return code: {ret}")

    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        MODEL_PATH = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} [model_path]")
        print("Using default paths")

    exit_code = aid_sdk_demo(MODEL_PATH)
    sys.exit(exit_code)