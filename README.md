# 🖐️ Gesture Volume Simulator

A **Streamlit-based web application** that uses computer vision to simulate volume control using hand gestures.  
When you hold your **thumb and index finger** together for 2 seconds, a virtual volume bar is activated.

---

## 🔍 Features

- Real-time webcam capture
- Detects thumb and index fingertip using MediaPipe
- Activates "volume control" after holding gesture for 2 seconds
- Simulated volume bar (no real system audio changes)

---

## 📦 Requirements

- Python 3.8+
- Streamlit
- OpenCV
- MediaPipe
- NumPy

Install all dependencies:

```bash
pip install -r requirements.txt
