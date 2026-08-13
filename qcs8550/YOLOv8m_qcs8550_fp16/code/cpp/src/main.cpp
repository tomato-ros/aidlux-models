#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "aidlite.hpp"

using namespace Aidlux::Aidlite;

std::string model_path = "../../models/QCS8550/FP16/yolov8m_qcs8550_fp16.qnn236.ctx.bin";
std::string image_path = "data/bus.jpg";
std::string result_path = "data/result.jpg";

const uint32_t width = 640;
const uint32_t height = 640;

struct Box {
    float x, y, w, h;
    float conf;
    int cls;
};

/**
 * @brief 和Python代码完全对齐图像预处理
 * @param image 输入OpenCV BGR Mat
 * @param input_size 目标正方形尺寸(640)
 * @return tuple{float数组(NHWC [1,H,W,3]), scale系数}
 */
std::tuple<std::vector<float>, float> preprocess_image(const cv::Mat &image, int input_size = 640) {
    int h = image.rows;
    int w = image.cols;
    int length = std::max(h, w);
    float scale = static_cast<float>(length) / input_size;

    cv::Mat canvas = cv::Mat::zeros(length, length, CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    image.copyTo(canvas(roi));

    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    cv::resize(canvas, canvas, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);

    std::vector<float> input_data(input_size * input_size * 3);
    float *dst_ptr = input_data.data();

    for (int y = 0; y < input_size; ++y) {
        const uint8_t *row_ptr = canvas.ptr<uint8_t>(y);
        for (int x = 0; x < input_size; ++x) {
            *dst_ptr++ = row_ptr[0] / 255.0f;
            *dst_ptr++ = row_ptr[1] / 255.0f;
            *dst_ptr++ = row_ptr[2] / 255.0f;
            row_ptr += 3;
        }
    }
    return {std::move(input_data), scale};
}

struct Tensor {
    void *data = nullptr;
    std::vector<uint64_t> shape;
    uint64_t elem_count = 0;
    Aidlux::Aidlite::DataType dtype;
};

std::vector<Tensor> fetch_all_outputs(
    Aidlux::Aidlite::Interpreter *interpreter,
    const std::vector<std::vector<uint64_t> > &output_shapes,
    bool is_native = false) {
    std::vector<Tensor> outputs;
    outputs.reserve(output_shapes.size());

    for (uint32_t i = 0; i < output_shapes.size(); ++i) {
        Tensor tensor;
        tensor.shape = output_shapes[i];

        uint64_t elem_cnt = 1;
        for (auto s: tensor.shape) elem_cnt *= s;
        tensor.elem_count = elem_cnt;

        void *out_ptr = nullptr;
        uint32_t raw_len = 0;
        int32_t ret = interpreter->get_output_tensor(
            i,
            &out_ptr,
            is_native,
            &raw_len
        );

        tensor.data = (ret == 0) ? out_ptr : nullptr;
        tensor.dtype = DataType::TYPE_FLOAT32;
        outputs.push_back(std::move(tensor));
    }
    return outputs;
}

/**
 * @brief 推理主逻辑，成功返回true，失败false
 */
bool run_inference(Interpreter *fast_interpreter) {
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        printf("imread failed\n");
        return false;
    }
    auto [input_tensor_data, scale] = preprocess_image(frame, width);

    int result = fast_interpreter->set_input_tensor(0, input_tensor_data.data(), false);
    if (result != 0) {
        printf("set_input_tensor failed ret=%d\n", result);
        return false;
    }

    // 性能测试，重复调用
    for (int i = 0; i < 100; i++) {
        result = fast_interpreter->invoke();
        if (result != 0) {
            printf("invoke failed ret=%d\n", result);
            return false;
        }
    }

    std::vector<std::vector<uint64_t> > output_shapes = {{1, 84, 8400}};
    auto outputs = fetch_all_outputs(fast_interpreter, output_shapes, false);

    if (outputs.empty() || outputs[0].data == nullptr) {
        printf("get output tensor failed\n");
        return false;
    }

    auto pred = reinterpret_cast<float *>(outputs[0].data);

    const int num_anchor = 8400;
    const int dim = 84;
    const float conf_thresh = 0.875f;
    std::vector<Box> boxes;

    // ===================== 适配 [1,84,8400] NC排布解码 =====================
    for (int idx = 0; idx < num_anchor; idx++) {
        float x = pred[0 * num_anchor + idx];
        float y = pred[1 * num_anchor + idx];
        float w = pred[2 * num_anchor + idx];
        float h = pred[3 * num_anchor + idx];

        // 遍历所有类别，找最高分和类别ID
        float max_cls_score = 0.f;
        int max_cls_id = 0;
        for (int c = 4; c < dim; c++) {
            // c=4开始：cls0 ~ cls79
            float score = pred[c * num_anchor + idx];
            if (score > max_cls_score) {
                max_cls_score = score;
                max_cls_id = c - 4;
            }
        }

        // 使用最大类别得分作为置信度阈值
        if (max_cls_score < conf_thresh)
            continue;

        Box b;
        b.x = x * scale;
        b.y = y * scale;
        b.w = w * scale;
        b.h = h * scale;
        b.conf = max_cls_score;
        b.cls = max_cls_id;
        boxes.push_back(b);
    }

    printf("detected box num: %zu\n", boxes.size());
    for (auto &b: boxes) {
        printf("cls:%d conf:%.3f x1:%.1f y1:%.1f w:%.1f h:%.1f\n", b.cls, b.conf, b.x, b.y, b.w, b.h);
        // 绘图
        cv::Rect2f rect(b.x - b.w / 2, b.y - b.h / 2, b.w, b.h);
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(result_path, frame);

    printf("save %s done\n", result_path.c_str());

    return true;
}

int aid_sdk_demo() {
    printf("Aidlite library version : %s\n", get_library_version().c_str());
    set_log_level(INFO);

    // 创建模型
    Model *model = Model::create_instance(model_path);
    if (model == nullptr) {
        printf("Create model failed !\n");
        return EXIT_FAILURE;
    }

    // !!!!!!!!!重点：当前模型输出 [1,84,8400] NC布局!!!!!!!!
    std::vector<std::vector<uint32_t> > input_shapes = {{1, height, width, 3}};
    std::vector<std::vector<uint32_t> > output_shapes_model = {{1, 84, 8400}};
    model->set_model_properties(input_shapes, DataType::TYPE_FLOAT32,
                                output_shapes_model, DataType::TYPE_FLOAT32);

    Config *config = Config::create_instance();
    if (config == nullptr) {
        printf("Create config failed !\n");
        return EXIT_FAILURE;
    }

    config->framework_type = FrameworkType::TYPE_QNN248;
    config->accelerate_type = AccelerateType::TYPE_DSP;
    config->is_quantify_model = 1;

    std::unique_ptr<Interpreter> fast_interpreter = InterpreterBuilder::build_interpreter_from_model_and_config(
        model, config);

    if (!fast_interpreter) {
        printf("build_interpreter_from_model_and_config failed !\n");
        return EXIT_FAILURE;
    }

    int result = fast_interpreter->init();
    if (result != 0) {
        printf("interpreter->init() failed ! ret=%d\n", result);
    } else {
        fast_interpreter->load_model();

        // 执行推理
        run_inference(fast_interpreter.get());
    }

    // 统一释放资源
    fast_interpreter->destroy();

    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    if (argc >= 4) {
        model_path = argv[1];
        image_path = argv[2];
        result_path = argv[3];
    } else if (argc != 1) {
        std::cout << "Usage: " << argv[0] << " [model_path] [image_path] [result_path]" << std::endl;
        std::cout << "Using default paths" << std::endl;
    }

    std::cout << "model_path " << model_path << std::endl;
    std::cout << "image_path " << image_path << std::endl;
    std::cout << "result_path " << result_path << std::endl;

    return aid_sdk_demo();
}
