// Version: Rev.D

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

    // --- PROPORTIONAL LETTERBOXING (RGB) ---
    // Calculate aspect-ratio preserving scale factor
    float scale = std::min(static_cast<float>(m_net_w) / orig_w, static_cast<float>(m_net_h) / orig_h);
    int new_w = static_cast<int>(std::round(orig_w * scale));
    int new_h = static_cast<int>(std::round(orig_h * scale));

    // Calculate padding required to reach 640x640 canvas (140 pixels top/bottom for 16:9)
    int pad_w = (m_net_w - new_w) / 2;
    int pad_h = (m_net_h - new_h) / 2;

    // Resize camera frame proportionally
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(new_w, new_h));

    // Create 640x640 canvas pre-filled with neutral YOLO gray (114, 114, 114)
    cv::Mat letterboxed(m_net_h, m_net_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_frame.copyTo(letterboxed(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // Correct color blindness (Convert BGR to RGB)
    cv::Mat rgb_frame;
    cv::cvtColor(letterboxed, rgb_frame, cv::COLOR_BGR2RGB);

    // Copy to persistent class buffer to prevent destruction
    std::memcpy(m_input_buffer.data(), rgb_frame.data, m_input_frame_size);

    m_bindings.input()->set_buffer(MemoryView(m_input_buffer.data(), m_input_frame_size));
    m_bindings.output()->set_buffer(MemoryView(m_output_buffer.data(), m_output_buffer.size()));

    // 2. Push to NPU and define the background callback
    auto status = m_configured_model.wait_for_async_ready(std::chrono::milliseconds(0));
    if (status == HAILO_SUCCESS) {
        
        m_configured_model.run_async(m_bindings, [this, orig_w, orig_h, pad_w, pad_h, scale, conf_threshold](const AsyncInferCompletionInfo& info) {
            
            size_t output_size = m_output_buffer.size();
            float* raw_data = reinterpret_cast<float*>(m_output_buffer.data());
            
            // Calculate how many maximum candidate boxes the NPU returns per class
            size_t total_floats = output_size / sizeof(float);
            size_t max_boxes_per_class = total_floats / (80 * 5); 

            std::vector<DetectionBox> temp_detections;

            // Mirror the Python tensor extraction: [80 classes][max_boxes][5 attributes]
            for (int cls_id = 0; cls_id < 80; ++cls_id) {
                for (size_t d = 0; d < max_boxes_per_class; ++d) {
        
                    // Calculate the exact memory address for this specific detection
                    size_t base_idx = (cls_id * max_boxes_per_class + d) * 5;
                    
                    float conf = raw_data[base_idx + 4];

                    // Only process valid boxes for Class 0 (Person)
                    if (conf >= conf_threshold && cls_id == 0) {
                        
                        float ymin_canvas = raw_data[base_idx + 0] * m_net_h;
                        float xmin_canvas = raw_data[base_idx + 1] * m_net_w;
                        float ymax_canvas = raw_data[base_idx + 2] * m_net_h;
                        float xmax_canvas = raw_data[base_idx + 3] * m_net_w;

                        // --- CENTROID MASKING FILTER ---
                        float centroid_y = (ymin_canvas + ymax_canvas) / 2.0f;
                        if (centroid_y <= pad_h || centroid_y >= (m_net_h - pad_h)) {
                            continue; 
                        }

                        // Reverse padding offsets and unscale
                        int x1 = static_cast<int>(std::round((xmin_canvas - pad_w) / scale));
                        int y1 = static_cast<int>(std::round((ymin_canvas - pad_h) / scale));
                        int x2 = static_cast<int>(std::round((xmax_canvas - pad_w) / scale));
                        int y2 = static_cast<int>(std::round((ymax_canvas - pad_h) / scale));

                        // Clamp coordinates to bounds
                        DetectionBox box;
                        box.x1 = std::max(0, std::min(x1, orig_w - 1));
                        box.y1 = std::max(0, std::min(y1, orig_h - 1));
                        box.x2 = std::max(0, std::min(x2, orig_w - 1));
                        box.y2 = std::max(0, std::min(y2, orig_h - 1));
                        box.confidence = conf;
                        box.class_id = cls_id;
                        temp_detections.push_back(box);
                    }
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