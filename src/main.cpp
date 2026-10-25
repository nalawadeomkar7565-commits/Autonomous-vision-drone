// Version: Rev.C
#include <iostream>
#include <opencv2/opencv.hpp>
#include "HailoYOLOv8.hpp"

int main() {
    float conf_th = 0.60f;

    // 1. Configure the Zero-Latency Camera Input (Forcing BGR format)
    std::string pipeline_in = "libcamerasrc ! videoconvert ! videoscale ! "
                              "video/x-raw, width=1024, height=576, framerate=30/1, format=BGR ! "
                              "appsink drop=true sync=false max-buffers=1";

    cv::VideoCapture cap(pipeline_in, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open camera." << std::endl;
        return -1;
    }

    // 2. Configure the RTSP Network Output
    std::string pipeline_out = "appsrc ! videoconvert ! video/x-raw, format=I420 ! "
                            "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast key-int-max=30 bframes=0 ! "
                            "rtspclientsink location=rtsp://127.0.0.1:8554/drone protocols=tcp";
    
    cv::VideoWriter out(pipeline_out, cv::CAP_GSTREAMER, 0, 30.0, cv::Size(1024, 576), true);
    if (!out.isOpened()) {
        std::cerr << "[ERROR] Failed to open UDP network stream." << std::endl;
        return -1;
    }

    // 3. Initialize the Hardware Engine
    std::string hef_path = "/usr/share/hailo-models/yolov8s_h8l.hef"; 
    std::cout << "[INFO] Spinning up Hailo NPU..." << std::endl;
    HailoYOLOv8 yolo(hef_path);

    cv::Mat frame;
    std::cout << "[INFO] Broadcasting live HUD via RTSP to 127.0.0.1:8554..." << std::endl;

    // 4. The Continuous Execution Loop
    while (true) {
        cap.read(frame);
        if (frame.empty()) continue;

        yolo.infer_async(frame, conf_th);
        std::vector<DetectionBox> detections = yolo.get_latest_detections();

        // HUD Overlays
        int center_x = frame.cols / 2;
        int center_y = frame.rows / 2;
        cv::line(frame, cv::Point(center_x - 20, center_y), cv::Point(center_x + 20, center_y), cv::Scalar(0, 255, 255), 2);
        cv::line(frame, cv::Point(center_x, center_y - 20), cv::Point(center_x, center_y + 20), cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, "SYSTEM: ARMED", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "TARGETING: STANDBY", cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        
        int target_counter = 1; // Resets for every new frame
        
        for (const auto& box : detections) {
            cv::rectangle(frame, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);
            
            // Append the counter to the display label
            std::string label = "Target " + std::to_string(target_counter) + ": " + std::to_string(box.confidence).substr(0, 4);
            cv::putText(frame, label, cv::Point(box.x1, box.y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            target_counter++; // Increment for the next object found in this frame
        }

        // Blast the frame over Wi-Fi instead of drawing locally
        out.write(frame);
    }

    cap.release();
    out.release();
    return 0;
}