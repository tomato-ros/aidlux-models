/*
 *  Copyright (c) 2025 Aplux Intelligence Technologies, Ltd.
 *  All rights reserved.
 *
 *  Confidential and Proprietary - Aplux Intelligence Technologies, Ltd.
 */
 
#include "qwen3vl_image_process.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

static bool load_bin_data(const std::string& filepath, std::vector<float>& out_data) {
  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "[ERROR] load_bin_data: cannot open " << filepath << std::endl;
    return false;
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  out_data.resize(size / sizeof(float));
  file.read(reinterpret_cast<char*>(out_data.data()), size);
  return !file.fail();
}

Qwen3VlVisionEncoder::Qwen3VlVisionEncoder(QwenVisionConfig& config)
  : Qwen3VlVisionEncoder(config.img_h, 
                           config.img_w, 
                           config.img_c, 
                           config.T, 
                           config.patch_size, 
                           config.temporal_patch_size, 
                           config.merge_size) {}

Qwen3VlVisionEncoder::Qwen3VlVisionEncoder(size_t img_h,
                                               size_t img_w,
                                               size_t img_c,
                                               size_t T,
                                               size_t patch_size,
                                               size_t temporal_patch_size,
                                               size_t merge_size)
: m_img_h(img_h),
  m_img_w(img_w),
  m_img_c(img_c),
  m_T(T),
  m_patch_size(patch_size),
  m_temporal_patch_size(temporal_patch_size),
  m_merge_size(merge_size) {

  m_img_area = m_img_h * m_img_w;
  m_img_elem_size = m_img_area * m_img_c;
  m_grid_step = m_patch_size * m_merge_size;

  chw_data.resize(m_img_elem_size);
  cthw_data.resize(m_img_elem_size * m_T);
  input_data.resize(m_img_elem_size * m_T);
}

Qwen3VlVisionEncoder::~Qwen3VlVisionEncoder() {
  if (m_interpreter) {
    m_interpreter->destory();
    m_interpreter.reset();
  }
  
}

bool Qwen3VlVisionEncoder::load(const Qwen3VlPath& path, const Aidlux::Aidlite::FrameworkType backend_type){
  return Qwen3VlVisionEncoder::load(path.model_path, path.pos_embed_cos_path, path.pos_embed_sin_path, path.window_attention_mask_path, path.full_attention_mask_path, backend_type);
}

bool Qwen3VlVisionEncoder::load(const std::string& model_path,
                                const std::string& pos_embed_cos_path,
                                const std::string& pos_embed_sin_path,
                                const std::string& window_attention_mask_path,
                                const std::string& full_attention_mask_path,
                                const Aidlux::Aidlite::FrameworkType backend_type) {
  Aidlux::Aidlite::Model* model = Aidlux::Aidlite::Model::create_instance(model_path.c_str());
  Aidlux::Aidlite::Config* config = Aidlux::Aidlite::Config::create_instance();
  config->implement_type = Aidlux::Aidlite::ImplementType::TYPE_LOCAL;
  config->framework_type = backend_type;
  config->accelerate_type = Aidlux::Aidlite::AccelerateType::TYPE_DSP;
  config->qnn_shared_buffer = 1;
  m_interpreter = Aidlux::Aidlite::InterpreterBuilder::build_interpretper_from_model_and_config(model, config);

  if (m_interpreter->init() < 0 || m_interpreter->load_model() < 0) {
    std::cerr << "[ERROR] load: QNN init/load_model failed." << std::endl;
    return false;
  }

  std::vector<std::vector<Aidlux::Aidlite::TensorInfo>> in_tensors;
  m_interpreter->get_input_tensor_info(in_tensors);
  for (const auto& tensor : in_tensors[0]) {
    if (tensor.name == "pixel_values" && !validate_model(tensor)) return false;
  }
  
  std::vector<std::vector<Aidlux::Aidlite::TensorInfo>> out_tensors;
  m_interpreter->get_output_tensor_info(out_tensors);
  m_img_pad_size = out_tensors[0][0].shape[0];
  m_embedding_size = out_tensors[0][0].shape[1];

  std::vector<float> cos_data, sin_data, win_mask, full_mask;
  if (!load_bin_data(pos_embed_cos_path, cos_data) ||
    !load_bin_data(pos_embed_sin_path, sin_data) ||
    !load_bin_data(window_attention_mask_path, win_mask) ||
    !load_bin_data(full_attention_mask_path, full_mask)) {
    std::cerr << "[ERROR] load: failed to load weight files." << std::endl;
    return false;
  }
  m_interpreter->set_input_tensor("position_ids_cos", cos_data.data());
  m_interpreter->set_input_tensor("position_ids_sin", sin_data.data());
  m_interpreter->set_input_tensor("window_attention_mask", win_mask.data());
  m_interpreter->set_input_tensor("full_attention_mask", full_mask.data());

  m_available = true;
  return true;
}

bool Qwen3VlVisionEncoder::encode(const cv::Mat& img_data, std::vector<uint8_t>& embedding) {
  using clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::duration<double, std::milli>;

  if (!m_available || !validate_image(img_data)) return false;

  // ── Stage 1: preprocess + set_input ──
  if (!preprocess(img_data, input_data)) {
    std::cerr << "[ERROR] encode: preprocess failed." << std::endl;
    return false;
  }
  m_interpreter->set_input_tensor("pixel_values", input_data.data());

  // ── Stage 2: invoke ──
  if (m_interpreter->invoke() < 0) {
    std::cerr << "[ERROR] encode: invoke failed." << std::endl;
    return false;
  }

  // ── Stage 3: get_output + copy ──
  float* buf = nullptr;
  uint32_t len = 0;
  m_interpreter->get_output_tensor(3, reinterpret_cast<void**>(&buf), &len);
  if (!buf || !len) {
    std::cerr << "[ERROR] encode: empty output tensor." << std::endl;
    return false;
  }
  embedding.resize(len);
  std::memcpy(embedding.data(), buf, len);
  return true;
}

bool Qwen3VlVisionEncoder::validate_model(const Aidlux::Aidlite::TensorInfo& tensor) {
  int num_patches = 0, patch_area = 0, valid_dim_count = 0;
  for (auto s : tensor.shape) {
    if (s > 1) {
      if (valid_dim_count == 0) num_patches = s;
      else if (valid_dim_count == 1) patch_area = s;
      valid_dim_count++;
    }
  }

  int expected_num_patches = m_img_area / (m_patch_size * m_patch_size);
  int expected_patch_area = m_img_c * m_T * m_patch_size * m_patch_size;

  if (num_patches != expected_num_patches || patch_area != expected_patch_area) {
    std::cerr << "[ERROR] Tensor 'pixel_values' shape mismatch!\n"
              << "  [User Parameters]\n"
              << "    - m_img_area   = " << m_img_area << "\n"
              << "    - m_patch_size = " << m_patch_size << "\n"
              << "    - m_img_c      = " << m_img_c << "\n"
              << "    - m_T          = " << m_T << "\n"
              << "  [Comparison]\n"
              << "    - num_patches: actual = " << num_patches 
              << ", expected = " << expected_num_patches 
              << " (formula: m_img_area / (m_patch_size * m_patch_size))\n"
              << "    - patch_area:  actual = " << patch_area 
              << ", expected = " << expected_patch_area 
              << " (formula: m_img_c * m_T * m_patch_size * m_patch_size)\n";
    return false;
  }
  return true;
}

bool Qwen3VlVisionEncoder::validate_image(const cv::Mat& img) {
  if (img.empty()) {
    std::cerr << "[ERROR] validate_image: empty image." << std::endl;
    return false;
  }
  if (img.cols != static_cast<int>(m_img_w) ||
    img.rows != static_cast<int>(m_img_h) ||
    img.channels() != static_cast<int>(m_img_c)) {
    std::cerr << "[ERROR] validate_image: expected " << m_img_w << "x" << m_img_h
              << "x" << m_img_c << ", got " << img.cols << "x" << img.rows
              << "x" << img.channels() << std::endl;
    return false;
  }
  return true;
}

bool Qwen3VlVisionEncoder::preprocess(const cv::Mat& img, std::vector<float>& output) {
  return normalize_to_chw(img, chw_data) &&
          stack_frames(chw_data, cthw_data) &&
          spatio_temporal_patch(cthw_data, output);
}

bool Qwen3VlVisionEncoder::normalize_to_chw(const cv::Mat& input, std::vector<float>& chw_data) {
  for (int c = 0; c < m_img_c; ++c) {
    float* dst_ptr = chw_data.data() + c * m_img_area;
    for (int h = 0; h < m_img_h; ++h) {
      const uchar* row_ptr = input.ptr<uchar>(h);
      for (int w = 0; w < m_img_w; ++w) {
        float src_value = static_cast<float>(row_ptr[w * m_img_c + c]);
        *dst_ptr++ = src_value / 127.5f - 1.0f;
      }
    }
  }
  return true;
}
 

// 复制一份 frame，组成两帧，形状为 c,2,h,w 的数据。
bool Qwen3VlVisionEncoder::stack_frames(const std::vector<float>& frames, std::vector<float>& cthw_data) {
  for (int c = 0; c < m_img_c; ++c) {
      float* dst_ptr = cthw_data.data() + c * m_temporal_patch_size * m_img_area;
      for (int t = 0; t < m_temporal_patch_size; ++t) {
          std::memcpy(dst_ptr + t * m_img_area, frames.data() + c * m_img_area, m_img_area * sizeof(float));
      }
  }
  return true;
}

bool Qwen3VlVisionEncoder::spatio_temporal_patch(const std::vector<float>& cthw_data, std::vector<float>& output) {
  float* dst = output.data();
  const float* src = cthw_data.data();

  const size_t patch_bytes = m_patch_size * sizeof(float);
  const size_t ct_total = m_img_c * m_temporal_patch_size;

  for (size_t row_idx = 0; row_idx < m_img_h; row_idx += m_grid_step) {
    for (size_t col_idx = 0; col_idx < m_img_w; col_idx += m_grid_step) {

      for (size_t my = 0; my < m_merge_size; ++my) {
        const size_t patch_y = row_idx + my * m_patch_size;
        for (size_t mx = 0; mx < m_merge_size; ++mx) {
          const size_t patch_x = col_idx + mx * m_patch_size;
          const size_t spatial_offset = patch_y * m_img_w + patch_x;

          for (size_t ct = 0; ct < ct_total; ++ct) {
            const float* src_ptr = src + ct * m_img_area + spatial_offset;
            for (size_t r = 0; r < m_patch_size; ++r) { 
              std::memcpy(dst, src_ptr + r * m_img_w, patch_bytes);
              dst += m_patch_size;
            }
          }

        }
      }

    }
  }
  return true;
}
