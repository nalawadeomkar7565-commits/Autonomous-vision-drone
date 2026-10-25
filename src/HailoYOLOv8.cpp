#include "HailoYOLOv8.hpp"
#include <iostream>

using namespace hailort;

HailoYOLOv8::HailoYOLOv8(const std::string& hef_path) {
    std::cout << "[INFO] Initializing Hailo NPU in C++ with: " << hef_path << std::endl;

    // 1. Create Virtual Device (Scan PCIe and claim the Hailo AI Hat)
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        throw std::runtime_error("Failed to create Hailo VDevice");
    }
    m_vdevice = vdevice_exp.release();

    // 2. Load HEF Model into memory
    auto infer_model_exp = m_vdevice->create_infer_model(hef_path);
    if (!infer_model_exp) {
        throw std::runtime_error("Failed to load HEF model into Hailo VDevice");
    }
    m_infer_model = infer_model_exp.release();

    // Configure format to UINT8 (RGB) for input and FLOAT32 for output
    m_infer_model->output()->set_format_type(HAILO_FORMAT_TYPE_FLOAT32);

    // Extract expected input resolution (e.g., 640x640) and required buffer size
    auto shape = m_infer_model->input()->shape();
    m_net_h = shape.height;
    m_net_w = shape.width;
    m_input_frame_size = m_infer_model->input()->get_frame_size();

    // 3. Configure Model for silicon execution
    auto configured_exp = m_infer_model->configure();
    if (!configured_exp) {
        throw std::runtime_error("Failed to configure InferModel");
    }
    m_configured_model = configured_exp.release();

    // 4. Create Stream Bindings
    auto bindings_exp = m_configured_model.create_bindings();
    if (!bindings_exp) {
        throw std::runtime_error("Failed to create Hailo bindings");
    }
    m_bindings = bindings_exp.release();
}

HailoYOLOv8::~HailoYOLOv8() {
    // Memory and VDevice resources cleanly destructed via unique_ptr
}

std::vector<DetectionBox> HailoYOLOv8::infer(const cv::Mat& frame, float conf_threshold) {
    int orig_w = frame.cols;
    int orig_h = frame.rows;

    // Pre-processing: Resize 1024x576 frame to NPU's expected 640x640 resolution & convert BGR to RGB
    cv::Mat resized_frame, rgb_frame;
    cv::resize(frame, resized_frame, cv::Size(m_net_w, m_net_h));
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Bind input memory buffer
    m_bindings.input()->set_buffer(MemoryView(rgb_frame.data, m_input_frame_size));

    // Prepare output buffer allocation
    size_t output_size = m_infer_model->output()->get_frame_size();
    std::vector<uint8_t> output_buffer(output_size);
    m_bindings.output()->set_buffer(MemoryView(output_buffer.data(), output_size));

    // Run Asynchronous Hardware Inference & Wait for Completion
    auto status = m_configured_model.wait_for_async_ready(std::chrono::milliseconds(1000));
    if (status == HAILO_SUCCESS) {
        m_configured_model.run(m_bindings, [](const AsyncInferCompletionInfo& info) {});
    }

    // Parse Output Tensors
    std::vector<DetectionBox> detections;
    float* raw_data = reinterpret_cast<float*>(output_buffer.data());
    
    // Hailo NMS output structure: [ymin, xmin, ymax, xmax, conf, class_id]
    size_t num_detections = output_size / (6 * sizeof(float));

    for (size_t i = 0; i < num_detections; ++i) {
        float conf = raw_data[i * 6 + 4];
        int class_id = static_cast<int>(raw_data[i * 6 + 5]);

        // Filter for Person class (class_id == 0) above threshold
        if (conf >= conf_threshold && class_id == 0) {
            float ymin = raw_data[i * 6 + 0];
            float xmin = raw_data[i * 6 + 1];
            float ymax = raw_data[i * 6 + 2];
            float xmax = raw_data[i * 6 + 3];

            DetectionBox box;
            box.x1 = static_cast<int>(xmin * orig_w);
            box.y1 = static_cast<int>(ymin * orig_h);
            box.x2 = static_cast<int>(xmax * orig_w);
            box.y2 = static_cast<int>(ymax * orig_h);
            box.confidence = conf;
            box.class_id = class_id;

            detections.push_back(box);
        }
    }

    return detections;
}
