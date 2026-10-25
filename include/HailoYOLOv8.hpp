// Version: Rev.D

#ifndef HAILO_YOLOV8_HPP
#define HAILO_YOLOV8_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "hailo/hailort.hpp"

struct DetectionBox {
    int x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class HailoYOLOv8 {
public:
    HailoYOLOv8(const std::string& hef_path);
    ~HailoYOLOv8();

    // Trigger NPU (Returns immediately, does not block)
    void infer_async(const cv::Mat& frame, float conf_threshold = 0.60f);

    // Fetch the latest results from RAM whenever the flight controller needs them
    std::vector<DetectionBox> get_latest_detections();

    int getNetWidth() const { return m_net_w; }
    int getNetHeight() const { return m_net_h; }

private:
    std::unique_ptr<hailort::VDevice> m_vdevice;
    std::shared_ptr<hailort::InferModel> m_infer_model;
    hailort::ConfiguredInferModel m_configured_model;
    hailort::ConfiguredInferModel::Bindings m_bindings;

    int m_net_w{640};
    int m_net_h{640};
    size_t m_input_frame_size{0};

    // --- NEW ASYNC PIPELINE ARCHITECTURE --- //
    
    // Persistent memory buffers that stay alive in RAM
    std::vector<uint8_t> m_input_buffer;
    std::vector<uint8_t> m_output_buffer;

    // Thread-safe storage for the latest targets
    std::vector<DetectionBox> m_latest_detections;
    std::mutex m_detection_mutex;

    // Hardware status flag to prevent overlapping commands
    std::atomic<bool> m_is_processing{false};
};

#endif // HAILO_YOLOV8_HPP