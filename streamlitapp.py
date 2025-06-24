import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import math
import time

st.set_page_config(page_title="Gesture Volume Control", layout="wide")
st.title("🖐️ Gesture Volume Control Simulator (WebRTC)")

# Transformer Class
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.start_touch_time = 0
        self.volume_control_active = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define red color range for masking
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2
                centers.append((cx, cy))
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

        if len(centers) >= 2:
            x1, y1 = centers[0]
            x2, y2 = centers[1]
            length = math.hypot(x2 - x1, y2 - y1)

            cv2.circle(img, (x1, y1), 12, (255, 0, 255), -1)
            cv2.circle(img, (x2, y2), 12, (0, 255, 255), -1)

            if length < 40:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)
                cv2.putText(img, "🟢 Holding...", (200, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

                if self.start_touch_time == 0:
                    self.start_touch_time = time.time()
                elif time.time() - self.start_touch_time >= 2:
                    self.volume_control_active = True
            else:
                self.start_touch_time = 0
                self.volume_control_active = False

            if self.volume_control_active:
                vol_bar = np.interp(length, [20, 200], [400, 150])
                vol_percent = np.interp(length, [20, 200], [0, 100])

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.rectangle(img, (50, 150), (90, 400), (50, 50, 50), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (90, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(vol_percent)} %', (45, 440), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(img, "✅ Volume Control Activated", (140, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(img, "Gesture Volume Control", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)
        return img


# Launch webcam stream
webrtc_streamer(
    key="gesture",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
