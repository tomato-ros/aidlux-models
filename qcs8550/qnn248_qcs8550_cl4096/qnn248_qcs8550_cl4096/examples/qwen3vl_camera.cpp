/*
 * qwen3vl_camera.cpp
 *
 * 功能：从摄像头采集画面，调用 Qwen3-VL-8B（QNN248 / NPU）描述图片内容。
 *
 * 流程：
 *   摄像头帧(cv::VideoCapture)
 *     -> VIT 视觉编码器(aidlite, Qwen3VlVisionEncoder) 得到图像 embedding
 *     -> 文本 tokenizer + embedding 查表(aidgen)
 *     -> 拼接 [文本前缀emb | 图像emb | 文本后缀emb]
 *     -> aidgen Generator 流式生成中文描述
 *
 * 用法：
 *   ./qwen3vl_camera [选项]
 *     -c <path>  aidgen 配置文件        (默认 aidgen_config.json)
 *     -m <path>  VIT 视觉模型           (默认 qwen3-vl-8b-vit.serialized.bin.aidem)
 *     -d <id>    摄像头设备号           (默认 0，即 /dev/video0)
 *     -p <text>  提问内容               (默认 "使用中文介绍一下这张图片的内容")
 *     -i <file>  用图片文件代替摄像头（无摄像头时的自测模式）
 *     -s <file>  把每次采集的帧保存成图片（默认 camera_frame.png，设 "none" 关闭）
 *     -b <type>  后端 qnn236|qnn240|qnn248|default (默认 qnn248)
 *
 * 运行后：按回车拍摄并描述，输入 q 再回车退出。
 */

#include "aidlux/aidgen/aidgen.hpp"
#include "qwen3vl_image_process.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <regex>
#include <cstring>

using namespace aplux::aidgen;

// ----------------------------- 可配置项 -----------------------------

struct CameraAppConfig {
    std::string aidgen_config = "aidgen_config.json";                      // LLM 配置
    std::string vit_model     = "qwen3-vl-8b-vit.serialized.bin.aidem";    // VIT 模型
    std::string prompt        = "使用中文介绍一下这张图片的内容";            // 提问
    std::string image_file;                                               // 自测模式图片
    std::string save_frame    = "camera_frame.png";                       // 采集帧保存路径
    int         camera_id     = 2;                                        // 摄像头编号
    BackendType aidgen_backend  = BackendType::QNN248;                      // aidgen 后端
    Aidlux::Aidlite::FrameworkType aidlite_backend =
        Aidlux::Aidlite::FrameworkType::TYPE_QNN248;                      // aidlite 后端
};

// Tensor 的自定义释放函数（aidgen 要求传入函数指针类型的 deleter）
void free_tensor_data(uint8_t* p) {
    if (p) std::free(p);
}

// ----------------------------- 参数解析 -----------------------------

static void print_usage(const char* prog) {
    std::cerr <<
        "用法: " << prog << " [选项]\n"
        "  -c <path>  aidgen 配置文件     (默认 aidgen_config.json)\n"
        "  -m <path>  VIT 视觉模型        (默认 qwen3-vl-8b-vit.serialized.bin.aidem)\n"
        "  -d <id>    摄像头设备号        (默认 0)\n"
        "  -p <text>  提问内容            (默认: 使用中文介绍一下这张图片的内容)\n"
        "  -i <file>  用图片文件自测，不开摄像头\n"
        "  -s <file>  采集帧保存路径      (默认 camera_frame.png，none 关闭)\n"
        "  -b <type>  后端: qnn236 | qnn240 | qnn248 | default (默认 qnn248)\n";
}

static bool parse_args(int argc, char* argv[], CameraAppConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] 选项 " << name << " 缺少参数值" << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (opt == "-c")      { auto v = need_value("-c"); if (!v) return false; cfg.aidgen_config = v; }
        else if (opt == "-m") { auto v = need_value("-m"); if (!v) return false; cfg.vit_model = v; }
        else if (opt == "-d") { auto v = need_value("-d"); if (!v) return false; cfg.camera_id = std::atoi(v); }
        else if (opt == "-p") { auto v = need_value("-p"); if (!v) return false; cfg.prompt = v; }
        else if (opt == "-i") { auto v = need_value("-i"); if (!v) return false; cfg.image_file = v; }
        else if (opt == "-s") { auto v = need_value("-s"); if (!v) return false; cfg.save_frame = v; }
        else if (opt == "-b") {
            auto v = need_value("-b"); if (!v) return false;
            std::string b = v;
            if (b == "qnn236") {
                cfg.aidgen_backend = BackendType::QNN236;
                cfg.aidlite_backend = Aidlux::Aidlite::FrameworkType::TYPE_QNN236;
            } else if (b == "qnn240") {
                cfg.aidgen_backend = BackendType::QNN240;
                cfg.aidlite_backend = Aidlux::Aidlite::FrameworkType::TYPE_QNN240;
            } else if (b == "qnn248") {
                cfg.aidgen_backend = BackendType::QNN248;
                cfg.aidlite_backend = Aidlux::Aidlite::FrameworkType::TYPE_QNN248;
            } else if (b == "default" || b == "auto") {
                cfg.aidgen_backend = BackendType::DEFAULT;
                cfg.aidlite_backend = Aidlux::Aidlite::FrameworkType::TYPE_DEFAULT;
            } else {
                std::cerr << "[ERROR] 未知后端类型: " << b << std::endl;
                return false;
            }
        } else {
            std::cerr << "[ERROR] 未知选项: " << opt << std::endl;
            return false;
        }
    }
    return true;
}

// ----------------------------- aidgen 封装 -----------------------------

// 集中管理 aidgen 的 Context / Generator / Tokenizer 和 embedding 查找表
struct AidgenPipeline {
    std::shared_ptr<Context>   ctx;
    std::unique_ptr<Generator> generator;
    std::unique_ptr<Tokenizer> tokenizer;
    const char* embedding_lut = nullptr;   // embedding 查找表（权重）起始地址
    size_t      embedding_lut_size = 0;    // 查找表总字节数
    size_t      embedding_size = 0;        // 单个向量的维度（如 4096）

    bool init(const CameraAppConfig& cfg) {
        // 1) Context：负责加载配置、tokenizer、embedding 表等公共资源
        ContextProperties props;
        props.stream = true;
        props.enable_profiler = true;
        ctx = Context::create_instance(cfg.aidgen_config, props, cfg.aidgen_backend);
        if (!ctx || ctx->initialize() != ErrorCode::SUCCESS) {
            std::cerr << "[ERROR] aidgen Context 初始化失败，请检查配置文件: "
                      << cfg.aidgen_config << std::endl;
            return false;
        }

        // 2) 从 Context 状态里解析 embedding-size（单个 token 向量的维度）
        std::string ctx_config = ctx->get_config();
        std::smatch match;
        if (!std::regex_search(ctx_config, match,
                               std::regex(R"("embedding-size"\s*:\s*(\d+))"))) {
            std::cerr << "[ERROR] 未能从 Context 配置中解析 embedding-size" << std::endl;
            return false;
        }
        embedding_size = std::stoull(match[1].str());

        // 3) Generator：LLM 推理本体（加载 6 个 serialized.bin 分片，跑在 NPU 上）
        generator = Generator::create_instance(ctx);
        if (!generator || generator->initialize() != ErrorCode::SUCCESS) {
            std::cerr << "[ERROR] aidgen Generator 初始化失败" << std::endl;
            return false;
        }
        generator->set_property("stream", "1");

        // 4) 拿到 embedding 查找表指针，后面用它把 token id 翻译成向量
        generator->get_embedding_buff(embedding_lut, embedding_lut_size);
        if (!embedding_lut || embedding_lut_size == 0) {
            std::cerr << "[ERROR] 获取 embedding 查找表失败" << std::endl;
            return false;
        }

        // 5) Tokenizer：文本 <-> token id
        tokenizer = Tokenizer::create_instance(ctx);
        if (!tokenizer || tokenizer->initialize() != ErrorCode::SUCCESS) {
            std::cerr << "[ERROR] aidgen Tokenizer 初始化失败" << std::endl;
            return false;
        }
        return true;
    }

    // 把一段文本编码成 token，再逐 token 查表，拼成 embedding 字节流
    std::vector<uint8_t> text_to_embedding(const std::string& text) {
        std::vector<int32_t> tokens;
        tokenizer->encode(text, tokens);
        const size_t row_bytes = embedding_size * sizeof(float);
        std::vector<uint8_t> out(tokens.size() * row_bytes);
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::memcpy(out.data() + i * row_bytes,
                        embedding_lut + tokens[i] * row_bytes,
                        row_bytes);
        }
        return out;
    }
};

// ----------------------------- 图文联合推理 -----------------------------

// 把 [文本前缀 emb] + [图像 emb] + [文本后缀 emb] 拼成一条序列送入 Generator，
// 流式打印模型回答。
static bool describe_image(AidgenPipeline& pipe,
                           Qwen3VlVisionEncoder& encoder,
                           const cv::Mat& frame_rgb,
                           const std::string& prompt) {
    // 1) VIT 编码：图像 -> embedding（aidlite / NPU）
    std::vector<uint8_t> image_emb;
    if (!encoder.encode(frame_rgb, image_emb) || image_emb.empty()) {
        std::cerr << "[ERROR] 图像编码失败" << std::endl;
        return false;
    }

    // 2) 构造 Qwen3-VL 对话模板，图像内容放在 <|vision_start|> ... <|vision_end|> 之间
    std::string text_before =
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|vision_start|>";
    std::string text_after =
        "<|vision_end|><|im_end|>\n<|im_start|>assistant\n";

    std::vector<uint8_t> emb_before = pipe.text_to_embedding(text_before);
    std::vector<uint8_t> emb_after  = pipe.text_to_embedding(text_after);

    // 3) 按顺序拼接成一整块 embedding 张量
    Tensor combined;
    combined.size = emb_before.size() + image_emb.size() + emb_after.size();
    combined.data = std::unique_ptr<uint8_t[], TensorDeleter>(
        static_cast<uint8_t*>(std::malloc(combined.size)), free_tensor_data);
    if (!combined.data) {
        std::cerr << "[ERROR] embedding 张量内存分配失败" << std::endl;
        return false;
    }
    uint8_t* dst = combined.data.get();
    std::memcpy(dst, emb_before.data(), emb_before.size()); dst += emb_before.size();
    std::memcpy(dst, image_emb.data(),  image_emb.size());  dst += image_emb.size();
    std::memcpy(dst, emb_after.data(),  emb_after.size());

    // 4) 流式推理：回调里逐段打印生成的文本
    std::cout << "模型回答: ";
    ErrorCode ec = pipe.generator->run(combined,
        [](const GenEvent& event, void*) {
            std::printf("%s", event.text.c_str());
            std::fflush(stdout);
        });
    std::cout << std::endl;
    pipe.generator->reset();   // 清空 KV cache，避免影响下一次提问

    if (ec != ErrorCode::SUCCESS) {
        std::cerr << "[ERROR] 推理失败, error code = "
                  << static_cast<int>(ec) << std::endl;
        return false;
    }
    return true;
}

// ----------------------------- 摄像头 -----------------------------

// 打开摄像头；先尝试 V4L2，失败后退回 OpenCV 默认后端
static bool open_camera(int camera_id, cv::VideoCapture& cap) {
    cap.open(camera_id, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        cap.open(camera_id);
    }
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] 无法打开摄像头 /dev/video" << camera_id << std::endl;
        return false;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::cout << "摄像头已打开: /dev/video" << camera_id
              << " (" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << ")" << std::endl;
    return true;
}

// 连续读几帧再取最后一帧：排空缓冲区并让自动曝光稳定
static bool capture_frame(cv::VideoCapture& cap, cv::Mat& frame, int warmup = 5) {
    for (int i = 0; i < warmup; ++i) {
        if (!cap.read(frame)) {
            std::cerr << "[ERROR] 读取摄像头帧失败" << std::endl;
            return false;
        }
    }
    return !frame.empty();
}

// ----------------------------- 主流程 -----------------------------

int main(int argc, char* argv[]) {
    CameraAppConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // 1. 初始化 LLM 推理管线（aidgen / QNN248 / NPU）
    AidgenPipeline pipe;
    if (!pipe.init(cfg)) return EXIT_FAILURE;
    std::cout << "[OK] 语言模型加载完成 (embedding-size=" << pipe.embedding_size << ")" << std::endl;

    // 2. 初始化 VIT 视觉编码器（aidlite / QNN248 / NPU）
    //    Qwen3-VL 的输入规格：512x512，patch 16，时间维度 2 帧，merge 2
    QwenVisionConfig vision_cfg;
    vision_cfg.img_h               = 512;
    vision_cfg.img_w               = 512;
    vision_cfg.img_c               = 3;
    vision_cfg.T                   = 2;
    vision_cfg.patch_size          = 16;
    vision_cfg.temporal_patch_size = 2;
    vision_cfg.merge_size          = 2;
    Qwen3VlVisionEncoder encoder(vision_cfg);

    Qwen3VlPath vision_path;
    vision_path.model_path                = cfg.vit_model;
    vision_path.pos_embed_cos_path        = "position_ids_cos.raw";
    vision_path.pos_embed_sin_path        = "position_ids_sin.raw";
    vision_path.window_attention_mask_path = "window_attention_mask.raw";
    vision_path.full_attention_mask_path  = "full_attention_mask.raw";
    if (!encoder.load(vision_path, cfg.aidlite_backend)) {
        std::cerr << "[ERROR] VIT 视觉编码器加载失败，请检查模型文件: "
                  << cfg.vit_model << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "[OK] 视觉编码模型加载完成" << std::endl;

    // 3. 自测模式：直接用图片文件跑一遍，不打开摄像头
    if (!cfg.image_file.empty()) {
        cv::Mat frame = cv::imread(cfg.image_file);
        if (frame.empty()) {
            std::cerr << "[ERROR] 无法读取图片: " << cfg.image_file << std::endl;
            return EXIT_FAILURE;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(vision_cfg.img_w, vision_cfg.img_h),
                   0.f, 0.f, cv::INTER_CUBIC);
        bool ok = describe_image(pipe, encoder, frame, cfg.prompt);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // 4. 摄像头模式：每次回车拍一张并描述
    cv::VideoCapture cap;
    if (!open_camera(cfg.camera_id, cap)) return EXIT_FAILURE;

    std::cout << "\n按回车拍摄并描述画面，输入 q 再回车退出。" << std::endl;
    std::string line;
    while (true) {
        std::cout << "\n[camera] 等待指令 >>> " << std::flush;
        if (!std::getline(std::cin, line)) break;          // EOF 也退出
        if (line == "q" || line == "Q") break;

        cv::Mat frame;
        if (!capture_frame(cap, frame)) continue;

        if (cfg.save_frame != "none") {
            cv::imwrite(cfg.save_frame, frame);            // 保存原始 BGR 帧
            std::cout << "[camera] 已保存采集帧到 " << cfg.save_frame << std::endl;
        }

        // OpenCV 读出来是 BGR、任意分辨率；模型需要 RGB、512x512
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        cv::resize(frame, frame, cv::Size(vision_cfg.img_w, vision_cfg.img_h),
                   0.f, 0.f, cv::INTER_CUBIC);

        describe_image(pipe, encoder, frame, cfg.prompt);
    }

    cap.release();
    std::cout << "退出。" << std::endl;
    return EXIT_SUCCESS;
}