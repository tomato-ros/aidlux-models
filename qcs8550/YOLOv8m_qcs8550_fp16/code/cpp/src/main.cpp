#include <iostream>

#include "aidlite.hpp"

using namespace Aidlux::Aidlite;

const std::string model_path = "../../models/QCS8550/FP16/yolov8m_qcs8550_fp16.qnn236.ctx.bin";
const std::string image_path = "data/bus.jpg";

const uint32_t width = 640;
const uint32_t height = 640;

int aid_sdk_demo() {

    // 获取SDK版本信息，设置日志相关事项
    printf("Aidlite library version : %s\n", get_library_version().c_str());

    set_log_level(WARNING);

    // 创建 Model 实例，并设置模型相关参数
    Model *model = Model::create_instance(model_path);

    if (model == nullptr) {

        printf("Create model failed !\n");

        return EXIT_FAILURE;
    }

    // 设置模型输入输出形状
    std::vector<std::vector<uint32_t> > input_shapes = {{1, height, width, 3}};
    std::vector<std::vector<uint32_t> > output_shapes = {{1, 84, 8400}};
    model->set_model_properties(input_shapes, DataType::TYPE_FLOAT32,
                                output_shapes, DataType::TYPE_FLOAT32);

    // 创建Config实例对象，并设置配置信息
    Config *config = Config::create_instance();

    if (config == nullptr) {

        printf("Create config failed !\n");

        return EXIT_FAILURE;
    }

    config->framework_type = FrameworkType::TYPE_QNN248;
    config->accelerate_type = AccelerateType::TYPE_DSP;
    config->is_quantify_model = 1;

    // 创建推理解释器对象
    std::unique_ptr<Interpreter> &&fast_interpreter = InterpreterBuilder::build_interpreter_from_model_and_config(model, config);

    if (fast_interpreter == nullptr) {

        printf("build_interpreter_from_model_and_config failed !\n");

        return EXIT_FAILURE;
    }

    // 完成解释器初始化
    int result = fast_interpreter->init();

    if (result != EXIT_SUCCESS) {

        printf("sample : interpreter->init() failed !\n");

        return EXIT_FAILURE;
    }

    // 加载模型
    fast_interpreter->load_model();

    //
    // //完整的推理代码示例：一般包括三部分：前处理 + 推理 + 后处理
    // {
    //     // 对于不同的模型，对应不同的前处理操作
    //     //void *input_tensor_data = preprocess();
    //     float input_tensor_data[512];
    //
    //     // 设置推理所需的输入数据
    //     result = fast_interpreter->set_input_tensor(0, input_tensor_data);
    //     if (result != EXIT_SUCCESS) {
    //         printf("sample : interpreter->set_input_tensor() failed !\n");
    //         return EXIT_FAILURE;
    //     }
    //     // 完成推理操作
    //     result = fast_interpreter->invoke();
    //     if (result != EXIT_SUCCESS) {
    //         printf("sample : interpreter->invoke() failed !\n");
    //         return EXIT_FAILURE;
    //     }
    //
    //     // 获取模型此次推理的结果数据
    //     float *out_data = nullptr;
    //     uint32_t output_tensor_length = 0;
    //     result = fast_interpreter->get_output_tensor(0, (void **) &out_data, &output_tensor_length);
    //     if (result != EXIT_SUCCESS) {
    //         printf("sample : interpreter->get_output_tensor() failed !\n");
    //         return EXIT_FAILURE;
    //     }
    //
    //     // 对于不同的模型，对应不同的后处理操作
    //     //int32_t result = postprocess(out_data);
    // }

    // 完成解释器资源释放操作
    result = fast_interpreter->destroy();

    if (result != EXIT_SUCCESS) {

        printf("interpreter->destroy() failed !\n");

        return EXIT_FAILURE;
    }
}

int main() {

    aid_sdk_demo();

    return 0;
}
