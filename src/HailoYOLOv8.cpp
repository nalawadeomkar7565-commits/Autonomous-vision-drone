// Version: Rev.B

#include "HailoYOLOv8.hpp"
#include <iostream>

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

    // Pre-allocate persistent RAM buffers once during startup
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

// Returns immediately. Does not freeze the CPU.
void HailoYOLOv8::infer_async(const cv::Mat& frame, float conf_threshold) {
    // 1. Zero-Latency Drop: If NPU is busy, discard this frame instantly
    if (m_is_processing.load()) {
        return; 
    }
    m_is_processing.store(true);

    int orig_w = frame.cols;
    int orig_h = frame.rows;

    cv::Mat resized_frame, rgb_frame;
    cv::resize(frame, resized_frame, cv::Size(m_net_w, m_net_h));
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Copy to persistent class buffer to prevent destruction
    std::memcpy(m_input_buffer.data(), rgb_frame.data, m_input_frame_size);

    m_bindings.input()->set_buffer(MemoryView(m_input_buffer.data(), m_input_frame_size));
    m_bindings.output()->set_buffer(MemoryView(m_output_buffer.data(), m_output_buffer.size()));

    // 2. Push to NPU and define the background callback
    auto status = m_configured_model.wait_for_async_ready(std::chrono::milliseconds(0));
    if (status == HAILO_SUCCESS) {
        
        // The lambda function [this, ...] captures class variables to use in the background
        m_configured_model.run(m_bindings, [this, orig_w, orig_h, conf_threshold](const AsyncInferCompletionInfo& info) {
            
            size_t output_size = m_output_buffer.size();
            float* raw_data = reinterpret_cast<float*>(m_output_buffer.data());
            size_t num_detections = output_size / (6 * sizeof(float));
            
            std::vector<DetectionBox> temp_detections;
            // Hailo NMS output structure: [ymin, xmin, ymax, xmax, conf, class_id]
            for (size_t i = 0; i < num_detections; ++i) {
                float conf = raw_data[i * 6 + 4];
                int class_id = static_cast<int>(raw_data[i * 6 + 5]);

                if (conf >= conf_threshold && class_id == 0) {
                    DetectionBox box;
                    box.x1 = static_cast<int>(raw_data[i * 6 + 1] * orig_w);
                    box.y1 = static_cast<int>(raw_data[i * 6 + 0] * orig_h);
                    box.x2 = static_cast<int>(raw_data[i * 6 + 3] * orig_w);
                    box.y2 = static_cast<int>(raw_data[i * 6 + 2] * orig_h);
                    box.confidence = conf;
                    box.class_id = class_id;
                    temp_detections.push_back(box);
                }
            }

            // 3. Lock memory and update the shared state
            {
                std::lock_guard<std::mutex> lock(m_detection_mutex);
                m_latest_detections = std::move(temp_detections);
            }

            // 4. Free the NPU for the next frame
            m_is_processing.store(false);
        });
    } else {
        m_is_processing.store(false); // Failsafe release
    }
}

// Allows the main loop to grab the newest targets without waiting
std::vector<DetectionBox> HailoYOLOv8::get_latest_detections() {
    std::lock_guard<std::mutex> lock(m_detection_mutex);
    return m_latest_detections;
}