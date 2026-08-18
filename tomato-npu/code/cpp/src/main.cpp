#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cinttypes>
#include <chrono>
#include <algorithm>

#include "aidlite.hpp"

using namespace Aidlux::Aidlite;

// ==================== 【配置常量区，统一维护】====================
std::string MODEL_PATH = "models/QCS8550/int8/libsimplenet_int8.so";

// SimpleNet 模型参数（与Python完全对齐）
constexpr uint32_t BATCH_SIZE = 1;
constexpr uint32_t INPUT_DIM = 16;
constexpr uint32_t OUTPUT_DIM = 4;

// Tensor索引
constexpr uint32_t INPUT_TENSOR_IDX = 0;
constexpr uint32_t OUTPUT_TENSOR_IDX = 0;

// INT8量化模型开关
constexpr bool IS_QUANT_MODEL = true;
// is_native=false：用户传float，SDK内部自动量化/反量化（推荐）
constexpr bool IS_NATIVE_DATA = false;

// 性能测试配置
constexpr int WARMUP_COUNT = 10; // 预热次数
constexpr int TEST_LOOP_COUNT = 50000; // 正式循环次数
// =================================================================

int aid_sdk_demo() {
    printf("Aidlite library version : %s\n", get_library_version().c_str());
    set_log_level(LogLevel::WARNING);

    // 1. 创建Model
    Model *model = Model::create_instance(MODEL_PATH);
    if (!model) {
        printf("[ERROR] Create model instance failed!\n");
        return EXIT_FAILURE;
    }

    std::vector<std::vector<uint32_t> > input_shapes = {{BATCH_SIZE, INPUT_DIM}};
    std::vector<std::vector<uint32_t> > output_shapes = {{BATCH_SIZE, OUTPUT_DIM}};

    DataType tensor_dtype = IS_QUANT_MODEL ? DataType::TYPE_INT8 : DataType::TYPE_FLOAT32;
    model->set_model_properties(input_shapes, tensor_dtype,
                                output_shapes, tensor_dtype);

    // 2. 创建Config
    Config *config = Config::create_instance();
    if (!config) {
        printf("[ERROR] Create config failed!\n");
        return EXIT_FAILURE;
    }
    config->framework_type = FrameworkType::TYPE_QNN;
    config->accelerate_type = AccelerateType::TYPE_DSP;
    config->is_quantify_model = IS_QUANT_MODEL ? 1 : 0;

    // 3. 构建解释器
    std::unique_ptr<Interpreter> fast_interpreter(
        InterpreterBuilder::build_interpreter_from_model_and_config(model, config)
    );
    if (!fast_interpreter) {
        printf("[ERROR] Build interpreter failed!\n");
        return EXIT_FAILURE;
    }

    // Model/Config构建完解释器即可释放
    config = nullptr;
    model = nullptr;

    // 4. 初始化 & 加载模型
    int32_t ret = fast_interpreter->init();
    if (ret != EXIT_SUCCESS) {
        printf("[ERROR] interpreter init failed, ret=%" PRId32 "\n", ret);
        fast_interpreter->destroy();
        return EXIT_FAILURE;
    }

    ret = fast_interpreter->load_model();
    if (ret != EXIT_SUCCESS) {
        printf("[ERROR] load_model failed, ret=%" PRId32 "\n", ret);
        fast_interpreter->destroy();
        return EXIT_FAILURE;
    }

    // ======================== 推理准备 ========================
    const size_t input_elem_cnt = BATCH_SIZE * INPUT_DIM;
    std::vector<float> input_fp32(input_elem_cnt, 0.1f);

    void *output_ptr = nullptr;
    uint32_t output_elem_num = 0;

    std::vector<double> invoke_cost_ms;
    invoke_cost_ms.reserve(TEST_LOOP_COUNT);

    // ========== 预热阶段 ==========
    printf("\n===== Warmup stage, iterations: %d =====\n", WARMUP_COUNT);
    for (int i = 0; i < WARMUP_COUNT; i++) {
        ret = fast_interpreter->set_input_tensor(INPUT_TENSOR_IDX, input_fp32.data(), IS_NATIVE_DATA);
        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] set_input_tensor failed, ret=%" PRId32 "\n", ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }
        ret = fast_interpreter->invoke();
        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] invoke failed at warmup, ret=%" PRId32 "\n", ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }
        ret = fast_interpreter->get_output_tensor(OUTPUT_TENSOR_IDX, &output_ptr, IS_NATIVE_DATA, &output_elem_num);
        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] get_output_tensor failed, ret=%" PRId32 "\n", ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }
    }
    printf("Warmup finished!\n\n");

    // ========== 正式循环压测 ==========
    printf("===== Start performance test, total loop: %d =====\n", TEST_LOOP_COUNT);
    for (int iter = 0; iter < TEST_LOOP_COUNT; iter++) {
        ret = fast_interpreter->set_input_tensor(INPUT_TENSOR_IDX, input_fp32.data(), IS_NATIVE_DATA);
        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] set_input_tensor failed, ret=%" PRId32 "\n", ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }

        // 只统计invoke推理耗时
        auto t_start = std::chrono::steady_clock::now();
        ret = fast_interpreter->invoke();
        auto t_end = std::chrono::steady_clock::now();

        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] invoke failed, iter=%d ret=%" PRId32 "\n", iter, ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }

        double cost_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        invoke_cost_ms.push_back(cost_ms);

        ret = fast_interpreter->get_output_tensor(OUTPUT_TENSOR_IDX, &output_ptr, IS_NATIVE_DATA, &output_elem_num);
        if (ret != EXIT_SUCCESS) {
            printf("[ERROR] get_output_tensor failed, ret=%" PRId32 "\n", ret);
            fast_interpreter->destroy();
            return EXIT_FAILURE;
        }

        // 每20轮打印一次输出，防止刷屏
        // if ((iter + 1) % 50 == 0) {
        //     printf("[%3d/%d] invoke cost: %.3f ms | Output: ", iter + 1, TEST_LOOP_COUNT, cost_ms);
        //     if (!IS_NATIVE_DATA) {
        //         float *out_fp32 = reinterpret_cast<float *>(output_ptr);
        //         for (uint32_t i = 0; i < output_elem_num; i++) {
        //             printf("%.4f ", out_fp32[i]);
        //         }
        //     }
        //     printf("\n");
        // }
    }

    // ========== 统计结果 ==========
    double sum_ms = 0.0;
    double max_ms = invoke_cost_ms[0];
    double min_ms = invoke_cost_ms[0];
    for (auto t: invoke_cost_ms) {
        sum_ms += t;
        max_ms = std::max(max_ms, t);
        min_ms = std::min(min_ms, t);
    }
    double avg_ms = sum_ms / TEST_LOOP_COUNT;

    printf("\n===== Test Result Summary =====\n");
    printf("Loop count      : %d\n", TEST_LOOP_COUNT);
    printf("Average invoke  : %.3f ms\n", avg_ms);
    printf("Min invoke time : %.3f ms\n", min_ms);
    printf("Max invoke time : %.3f ms\n", max_ms);
    printf("FPS estimate    : %.2f\n", 1000.0 / avg_ms);

    // 释放资源
    ret = fast_interpreter->destroy();
    if (ret != EXIT_SUCCESS) {
        printf("[WARN] interpreter destroy return code: %" PRId32 "\n", ret);
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {

    if (argc >= 1) {

        MODEL_PATH = argv[1];

    } else if (argc <= 0) {

        std::cout << "Usage: " << argv[0] << " [model_path]" << std::endl;

        std::cout << "Using default paths" << std::endl;
    }

    return aid_sdk_demo();
}