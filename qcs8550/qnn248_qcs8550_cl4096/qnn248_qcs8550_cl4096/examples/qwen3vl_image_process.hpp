/*
 *  Copyright (c) 2025 Aplux Intelligence Technologies, Ltd.
 *  All rights reserved.
 *
 *  Confidential and Proprietary - Aplux Intelligence Technologies, Ltd.
*/

#pragma once

#include "opencv2/opencv.hpp"
#include "aidlux/aidlite/aidlite.hpp"

#include <iostream>
#include <cstdint>
#include <memory>

struct QwenVisionConfig {
    size_t img_h;
    size_t img_w;
    size_t img_c;
    size_t T;
    size_t patch_size;
    size_t temporal_patch_size;
    size_t merge_size;
};

struct Qwen3VlPath {
  std::string model_path;
  std::string pos_embed_cos_path;
  std::string pos_embed_sin_path;
  std::string window_attention_mask_path;
  std::string full_attention_mask_path;
};

class Qwen3VlVisionEncoder{
public:
  Qwen3VlVisionEncoder(QwenVisionConfig& config);

  Qwen3VlVisionEncoder(size_t img_h,
                        size_t img_w,
                        size_t img_c,
                        size_t T,
                        size_t patch_size,
                        size_t temporal_patch_size,
                        size_t merge_size);
  ~Qwen3VlVisionEncoder();
  
  bool load(const Qwen3VlPath& path, const Aidlux::Aidlite::FrameworkType backend_type);

  bool load(const std::string& model_path,
            const std::string& pos_embed_cos_path,
            const std::string& pos_embed_sin_path,
            const std::string& window_attention_mask_path,
            const std::string& full_attention_mask_path,
            const Aidlux::Aidlite::FrameworkType backend_type);
  
  bool encode(const cv::Mat& img_data, std::vector<uint8_t>& embedding);

private:
  size_t m_img_h{0};
  size_t m_img_w{0};
  size_t m_img_c{0};
  size_t m_T{0};
  size_t m_patch_size{0};
  size_t m_temporal_patch_size{0};
  size_t m_merge_size{0};

  // 衍生参数
  size_t m_img_area{0};
  size_t m_img_elem_size{0};
  size_t m_grid_step{0};

  // 从模型获取的参数
  size_t m_img_pad_size{0};
  size_t m_embedding_size{0};

  bool m_available{false};

  // 预分配内存
  std::vector<float> chw_data;
  std::vector<float> cthw_data;
  std::vector<float> input_data;

  // 模型句柄
  std::unique_ptr<Aidlux::Aidlite::Interpreter> m_interpreter;

  bool validate_model(const Aidlux::Aidlite::TensorInfo& tensor);
  bool validate_image(const cv::Mat& img_data);

  bool preprocess(const cv::Mat& input_img, std::vector<float>& output);

  bool normalize_to_chw(const cv::Mat& input, std::vector<float>& chw_data);
  bool stack_frames(const std::vector<float>& frames, std::vector<float>& cthw_data);
  bool spatio_temporal_patch(const std::vector<float>& cthw_data, std::vector<float>& output);

};