// Version: Rev.C

#include "HailoYOLOv8.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace hailort;

HailoYOLOv8::HailoYOLOv8(const std::string& hef_path) {
    std::cout << "[INFO] Initializing Async Hailo NPU: " << hef_path << std::endl;

    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) throw std::runtime_error("Failed to create Hailo VDevice");
    m_vdevice = vdevice_exp.release();

    auto infer_model_exp = m_vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) throw std::runtime_error("Failed to load HEF");
    m_infer_model = infer_model_exp.release();

    m_infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);

    auto shape = m_infer_model->input()->shape();
    m_net_h = shape.height;
    m_net_w = shape.width;
    m_input_frame_size = m_infer_model->input()->get_frame_size();

    m_input_buffer.resize(m_input_frame_size);
    size_t output_size = m_infer_model->output()->get_frame_size();
    m_output_buffer.resize(output_size);

    auto configured_exp = m_infer_model->configure();
    if (!configured_exp) throw std::runtime_error("Failed to configure InferModel");
    m_configured_model = configured_exp.release();

    auto bindings_exp = m_configured_model.create_bindings();
    if (!bindings_exp) throw std::runtime_error("Failed to create Hailo bindings");
    m_bindings = bindings_exp.release();
}

HailoYOLOv8::~HailoYOLOv8() {}

void HailoYOLOv8::infer_async(const cv::Mat& frame, float conf_threshold) {
    if (m_is_processing.load()) {
        return; 
    }
    m_is_processing.store(true);

    int orig_w = frame.cols;
    int orig_h = frame.rows;

    // --- BGR SQUISH EXPERIMENT LOGIC ---
    // 1. Squish the 16:9 frame directly into the 1:1 NPU resolution (No padding)
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(m_net_w, m_net_h));

    // 2. Skip RGB conversion (Force the NPU to be color-blind like the Python script)
    // We copy the raw BGR data directly into the input buffer
    std::memcpy(m_input_buffer.data(), resized_frame.data, m_input_frame_size);

    m_bindings.input()->set_buffer(MemoryView(m_input_buffer.data(), m_input_frame_size));
    m_bindings.output()->set_buffer(MemoryView(m_output_buffer.data(), m_output_buffer.size()));

    auto status = m_configured_model.wait_for_async_ready(std::chrono::milliseconds(0));
    if (status == HAILO_SUCCESS) {
        
        m_configured_model.run_async(m_bindings, [this, orig_w, orig_h, conf_threshold](const AsyncInferCompletionInfo& info) {
            
            size_t output_size = m_output_buffer.size();
            float* raw_data = reinterpret_cast<float*>(m_output_buffer.data());
            size_t num_detections = output_size / (6 * sizeof(float));
            
            std::vector<DetectionBox> temp_detections;
            
            for (size_t i = 0; i < num_detections; ++i) {
                float conf = raw_data[i * 6 + 4];
                int class_id = static_cast<int>(raw_data[i * 6 + 5]);

                if (conf >= conf_threshold && class_id == 0) {
                    // Because we uniformly squished the image, we can multiply the normalized 
                    // NPU tensor outputs directly against the original 1024x576 dimensions.
                    int x1 = static_cast<int>(std::round(raw_data[i * 6 + 1] * orig_w));
                    int y1 = static_cast<int>(std::round(raw_data[i * 6 + 0] * orig_h));
                    int x2 = static_cast<int>(std::round(raw_data[i * 6 + 3] * orig_w));
                    int y2 = static_cast<int>(std::round(raw_data[i * 6 + 2] * orig_h));

                    DetectionBox box;
                    box.x1 = std::max(0, std::min(x1, orig_w - 1));
                    box.y1 = std::max(0, std::min(y1, orig_h - 1));
                    box.x2 = std::max(0, std::min(x2, orig_w - 1));
                    box.y2 = std::max(0, std::min(y2, orig_h - 1));
                    box.confidence = conf;
                    box.class_id = class_id;
                    temp_detections.push_back(box);
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_detection_mutex);
                m_latest_detections = std::move(temp_detections);
            }

            m_is_processing.store(false);
        });
    } else {
        m_is_processing.store(false);
    }
}

std::vector<DetectionBox> HailoYOLOv8::get_latest_detections() {
    std::lock_guard<std::mutex> lock(m_detection_mutex);
    return m_latest_detections;
}