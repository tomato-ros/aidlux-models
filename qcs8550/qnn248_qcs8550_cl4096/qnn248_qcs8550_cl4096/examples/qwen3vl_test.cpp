/*
 *  Copyright (c) 2025 Aplux Intelligence Technologies, Ltd.
 *  All rights reserved.
 *
 *  Confidential and Proprietary - Aplux Intelligence Technologies, Ltd.
 */
 
#include "aidlux/aidgen/aidgen.hpp"
#include "qwen3vl_image_process.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <regex>

using namespace aplux::aidgen;

void free_tensor_data(uint8_t* p) {
  if (p) std::free(p);
}

static std::string to_lower(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return r;
}

static bool parse_backend_type(const std::string& type_str, BackendType& aidgen_type, Aidlux::Aidlite::FrameworkType& aidlite_type) {
    std::string lower = to_lower(type_str);
    if (lower == "qnn236") { 
      aidgen_type  = BackendType::QNN236;
      aidlite_type = Aidlux::Aidlite::FrameworkType::TYPE_QNN236;
      return true; 
    } else if (lower == "qnn240") { 
      aidgen_type  = BackendType::QNN240;
      aidlite_type = Aidlux::Aidlite::FrameworkType::TYPE_QNN240;
      return true; 
    } else if (lower == "qnn248") { 
      aidgen_type  = BackendType::QNN248;
      aidlite_type = Aidlux::Aidlite::FrameworkType::TYPE_QNN248;
      return true; 
    } else if (lower == "default" || lower == "auto") {
      aidgen_type  = BackendType::DEFAULT;
      aidlite_type = Aidlux::Aidlite::FrameworkType::TYPE_DEFAULT;
      return true; 
    }
    return false;
}

static void print_usage(const char* prog_name) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << prog_name << " [backend_type]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Arguments:" << std::endl;
    std::cerr << "  backend_type  Backend version: qnn236 | qnn240 | qnn248 | default (optional, default: default)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Examples:" << std::endl;
    std::cerr << "  " << prog_name << " qnn236" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  BackendType aidgen_type = BackendType::DEFAULT;
  Aidlux::Aidlite::FrameworkType aidlite_type = Aidlux::Aidlite::FrameworkType::TYPE_DEFAULT;
  if (!parse_backend_type(argv[1], aidgen_type, aidlite_type)) {
      std::cerr << "Error: Unknown backend type '" << argv[1]
                << "'. Valid: qnn236, qnn240, qnn248, default" << std::endl;
      print_usage(argv[0]);
      return EXIT_FAILURE;
  }

  // 1. 创建并初始化 Context
  ContextProperties props;
  props.stream = false;
  props.enable_profiler = true;
  auto ctx = Context::create_instance("aidgen_config.json", props, aidgen_type);
  if (!ctx) return -1;
  if (ctx->initialize() != ErrorCode::SUCCESS) return -1;
  std::string ctx_config = ctx->get_config();
  std::cout<< ctx_config << std::endl;

  size_t embedding_size;
  std::regex reg(R"("embedding-size"\s*:\s*(\d+))");
  std::smatch match;
  if (std::regex_search(ctx_config, match, reg)) {
    embedding_size = std::stoull(match[1].str());
  } else {
    std::cerr << "未找到 embedding-size" << std::endl;
    return -1;
  }

  // 2. 创建并初始化模型
  QwenVisionConfig m_config;
  m_config.img_h               = 512;
  m_config.img_w               = 512;
  m_config.img_c               = 3;
  m_config.T                   = 2;
  m_config.patch_size          = 16;
  m_config.temporal_patch_size = 2;
  m_config.merge_size          = 2;
  Qwen3VlVisionEncoder m_encoder(m_config);

  Qwen3VlPath m_path;
  m_path.model_path = "qwen3-vl-8b-vit.serialized.bin.aidem";
  m_path.pos_embed_cos_path = "position_ids_cos.raw";
  m_path.pos_embed_sin_path = "position_ids_sin.raw";
  m_path.window_attention_mask_path = "window_attention_mask.raw";
  m_path.full_attention_mask_path = "full_attention_mask.raw";
  m_encoder.load(m_path, aidlite_type);
  std::cout<< "编码模型加载完成" << std::endl;
  
  
  auto generator = Generator::create_instance(ctx);
  if (!generator) return -1;
  if (generator->initialize() != ErrorCode::SUCCESS) return -1;
  std::cout<< generator->get_property() << std::endl;
  generator->set_property("stream", "1");

  const char* embedding_weights_buf = nullptr;
  size_t embedding_weights_size = 0;
  generator->get_embedding_buff(embedding_weights_buf, embedding_weights_size);

  auto tokenizer = Tokenizer::create_instance(ctx);
  if (!tokenizer) return -1;
  if (tokenizer->initialize() != ErrorCode::SUCCESS) return -1;
  std::cout<< "语言模型加载完成" << std::endl;
  

  // 3. 预设文本
  std::string test = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n介绍一下你自己./no_think<|im_end|>\n<|im_start|>assistant\n";
  
  cv::Mat frame = cv::imread("ingredients.png");
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  cv::resize(frame, frame, cv::Size(m_config.img_w , m_config.img_h), 0.f, 0.f, cv::INTER_CUBIC);

  std::cout<< "#################################################" << std::endl;
  // 4. tokenizer 测试
  std::cout<< "Tokenizer 编码测试: " << std::endl;
  std::vector<int32_t> tokens;
  tokenizer->encode(test, tokens);
  for(auto x : tokens) {
      std::cout<< x << " ";
  }
  std::cout<< std::endl;
    
  std::cout<< "Tokenizer 解码测试: " << std::endl;
  std::string dec_test;
  tokenizer->decode(tokens, dec_test);
  std::cout<< dec_test << std::endl;
  std::cout<<std::endl;
  std::cout<< "#################################################" << std::endl;

  // 5. 文本推理测试
  if (1) {
    std::cout<< "字符串输入" << std::endl;
    generator->run(test,
      [](const GenEvent& event, void*) {
          printf("%s", event.text.c_str());
          fflush(stdout);
      });
    generator->reset();
    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<< "#################################################" << std::endl;
  }

  // 6. 文本embedding推理测试
  if (1) {
    std::cout<< "Embedding输入" << std::endl;
    Tensor combined;
    combined.size = tokens.size() * embedding_size * sizeof(float);
    combined.data = std::unique_ptr<uint8_t[], TensorDeleter>(
        static_cast<uint8_t*>(std::malloc(combined.size)), 
        free_tensor_data
    );

    for (int i = 0; i < tokens.size(); i++) {
      std::memcpy((char*)combined.data.get() + i * embedding_size * sizeof(float), 
        embedding_weights_buf + tokens[i] * embedding_size * sizeof(float), 
        embedding_size * sizeof(float)
      );
    }

    generator->run(combined,
      [](const GenEvent& event, void*) {
          printf("%s", event.text.c_str());
          fflush(stdout);
      });

    generator->reset();
    std::cout<<std::endl;
    std::cout<< "#################################################" << std::endl;
  }

  // 7. 图文推理测试
  if (1) {
    std::cout<< "图文Embedding输入" << std::endl;
    std::string text_before = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n使用中文介绍一下这张图片的内容:<|vision_start|>";
    std::string text_after  = "<|vision_end|><|im_end|>\n<|im_start|>assistant\n";

    std::vector<int32_t> tokens_before;
    tokenizer->encode(text_before, tokens_before);

    std::vector<int32_t> tokens_after;
    tokenizer->encode(text_after, tokens_after);

    std::vector<uint8_t> m_result;
    m_encoder.encode(frame, m_result);

    Tensor combined;
    combined.size = tokens_before.size() * embedding_size * sizeof(float) + m_result.size() + tokens_after.size() * embedding_size * sizeof(float);
    combined.data = std::unique_ptr<uint8_t[], TensorDeleter>(
        static_cast<uint8_t*>(std::malloc(combined.size)), 
        free_tensor_data
    );

    size_t offset = 0;
    for (int i = 0; i < tokens_before.size(); i++) {
      std::memcpy(
        (char*)combined.data.get() + i * embedding_size * sizeof(float), 
        embedding_weights_buf + tokens_before[i] * embedding_size * sizeof(float), 
        embedding_size * sizeof(float)
      );
    }
    offset += tokens_before.size() * embedding_size * sizeof(float);

    std::memcpy((char*)combined.data.get() + offset, m_result.data(), m_result.size());
    offset += m_result.size();

    for (int i = 0; i < tokens_after.size(); i++) {
      std::memcpy(
        (char*)combined.data.get() + offset + i * embedding_size * sizeof(float), 
        embedding_weights_buf + tokens_after[i] * embedding_size * sizeof(float), 
        embedding_size * sizeof(float)
      );
    }

    generator->run(combined,
      [](const GenEvent& event, void*) {
          printf("%s", event.text.c_str());
          fflush(stdout);
      });

    generator->reset();
    std::cout<<std::endl;
    std::cout<< "#################################################" << std::endl;
  }
  return 0;
}
