import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer,WebRtcMode, RTCConfiguration,VideoProcessorBase

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

webrtc_streamer(
    key="object-detection",
    video_processor_factory=lambda: YOLOVideoProcessor(model),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=RTC_CONFIGURATION,
    mode=WebRtcMode.SENDRECV,
)


# Define the YOLO Model
model_path = r"C:\Users\shiva\Downloads\Blindman_project\final\yolo\runs\detect\Yolo11_new_results\weights\best.pt"
model = YOLO(model_path).to("cpu")

# Define the Video Processor Class for Live Detection
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.model = model

    def predict_and_detect(self, img, conf=0.5, rectangle_thickness=4, text_thickness=2):
        results = self.model.predict(img, conf=conf)
        for result in results:
            for box in result.boxes:
                # Draw rectangle
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                # Add label text
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img

    def recv(self, frame):
        # Convert frame from WebRTC format to OpenCV format
        img = frame.to_ndarray(format="bgr")
        img = self.predict_and_detect(img, conf=0.5)
        return img

# Streamlit UI Components
st.title("YOLO Webcam Object Detection")
st.write("This app uses YOLO to perform live object detection on webcam footage.")

# Start Webcam Stream with WebRTC
webrtc_streamer(
    key="object-detection",
    video_processor_factory=lambda: YOLOVideoProcessor(model),
    media_stream_constraints={"video": True, "audio": False}
)
