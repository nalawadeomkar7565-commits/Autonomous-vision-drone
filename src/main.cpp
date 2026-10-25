// Version: Rev.B

#include <iostream>
#include <opencv2/opencv.hpp>
#include "HailoYOLOv8.hpp"

int main() {
    // 1. Configure the Zero-Latency Camera Pipeline
    std::string pipeline = "libcamerasrc ! video/x-raw, width=1024, height=576, framerate=30/1 ! "
                           "videoconvert ! appsink drop=true max-buffers=1";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Failed to open GStreamer camera pipeline." << std::endl;
        return -1;
    }

    // 2. Initialize the Hardware Engine
    // Ensure this path points to where your .hef file is actually located on the Pi
    std::string hef_path = "../yolov8s_h8l.hef"; 
    
    std::cout << "[INFO] Spinning up Hailo NPU..." << std::endl;
    HailoYOLOv8 yolo(hef_path);

    cv::Mat frame;
    std::cout << "[INFO] Starting main flight loop..." << std::endl;

    // 3. The Continuous Execution Loop
    while (true) {
        // Pull the absolute freshest frame from the RAM mailbox
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "[WARNING] Camera dropped a frame." << std::endl;
            continue;
        }

        // Trigger the AI math in the background (returns instantly)
        yolo.infer_async(frame, 0.60f);

        // Retrieve the newest available coordinates (copies safely via mutex)
        std::vector<DetectionBox> detections = yolo.get_latest_detections();

        // 4. Visual Verification (HUD)
        for (const auto& box : detections) {
            // Draw target box
            cv::rectangle(frame, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);
            
            // Draw label and confidence score
            std::string label = "Target: " + std::to_string(box.confidence).substr(0, 4);
            cv::putText(frame, label, cv::Point(box.x1, box.y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        // Push the drawn frame to the screen
        cv::imshow("Drone Target Acquisition", frame);

        // Wait 1ms for a keystroke. If 'q' is pressed, break the loop and land.
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // 5. Clean Shutdown
    cap.release();
    cv::destroyAllWindows();
    std::cout << "[INFO] System safely shut down." << std::endl;
    return 0;
}
