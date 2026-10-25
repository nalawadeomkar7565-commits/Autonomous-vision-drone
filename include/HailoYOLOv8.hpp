#ifndef HAILO_YOLOV8_HPP // Header Guard to ensure that the header file is only copied into compiler unit once. 
#define HAILO_YOLOV8_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "hailo/hailort.hpp"

// Bounding box output contract
struct DetectionBox {
    int x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class HailoYOLOv8 {
public:
    HailoYOLOv8(const std::string& hef_path);
    ~HailoYOLOv8();

    // Main inference function
    std::vector<DetectionBox> infer(const cv::Mat& frame, float conf_threshold = 0.60f);

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
};

#endif // HAILO_YOLOV8_HPP // Wrapper for the Header Guard. 

/*
#ifndef "XYZ"
...
... Code here ...
...

#endif
*/
