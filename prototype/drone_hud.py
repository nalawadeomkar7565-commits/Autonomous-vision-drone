#Author: Omkar Nalawade
#Date: 26th September 2026
#Description:   Input and output pipeline for the Gstreamer and OpenCV to work together. 
#               And at the end overlay HUD information and stream it via RTSP.


#Update:                
#Update description:                     

#-----------------------------------------------------------------------------------

import cv2

# 1. Input: Capture the camera feed using the Pi's native libcamera system
cap_pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1, format=RGB ! videoconvert ! appsink"
cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)

# 2. Output: Encode the frames and push them to your MediaMTX server
out_pipeline = "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtspclientsink location=rtsp://127.0.0.1:8554/drone"
out = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480), True)

print("HUD Pipeline Active. Streaming to rtsp://dronepi.local:8554/drone")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- HUD OVERLAY LOGIC ---
    height, width, _ = frame.shape
    center_x, center_y = int(width / 2), int(height / 2)

    # Draw Target Crosshairs (Green)
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

    # Draw System Status Text (Red/Yellow)
    cv2.putText(frame, "SYSTEM: ARMED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "TARGETING: STANDBY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 3. Push the annotated frame to the RTSP server
    out.write(frame)

cap.release()
out.release()
