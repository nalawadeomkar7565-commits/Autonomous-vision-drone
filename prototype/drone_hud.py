#Author: Omkar Nalawade
#Date: 01st October 2026
#Description:   Input and output pipeline for the Gstreamer and OpenCV to work together. 
#               And at the end overlay HUD information and stream it via RTSP.

#Update:                Rev.A to Rev.B
#Update description:    Change in the captured video aspect ratio - 640x480 to 1024x576
#                       Output to RTSP bitrate change - 500 to 2000
#                       Adding variables

#-----------------------------------------------------------------------------------

import cv2

WIDTH = 1024
HEIGHT = 576
FPS = 30
BITRATE = 1500
IS_COLOUR = True

# 1. Input: Let libcamera capture natively, then scale and convert to OpenCV's native BGR format
cap_pipeline = (
    f"libcamerasrc ! "
    f"videoconvert ! "
    f"videoscale ! "
    f"video/x-raw, width={WIDTH}, height={HEIGHT}, framerate={FPS}/1, format=BGR ! "
    f"appsink drop=true sync=false"
)
cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)

# 2. Output: Added explicit Keyframe limits and disabled B-frames for zero-latency recovery
out_pipeline = (
    f"appsrc ! "
    f"videoconvert ! "
    f"video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate={BITRATE} speed-preset=superfast key-int-max={FPS} bframes=0 ! "
    f"rtspclientsink location=rtsp://127.0.0.1:8554/drone protocols=tcp"
)

out = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, float(FPS), (WIDTH, HEIGHT), IS_COLOUR)

if not out.isOpened():
    print("FATAL: Output pipeline failed to open!")
    exit()

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