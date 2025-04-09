import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar cascades
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Image processing functions
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast_brightness(image, contrast=1.0, brightness=0):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

# Get available cameras
def available_cameras(max_index=5):
    cams = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cams.append(i)
        cap.release()
    return cams

# Streamlit app UI
st.set_page_config(layout="wide")
st.title("📷 Face Detection with Camera Controls")

# Sidebar controls
st.sidebar.header("🔧 Settings")

# Camera selection
cam_options = available_cameras()
if not cam_options:
    st.error("No cameras found.")
    st.stop()

camera_index = st.sidebar.selectbox("Select Camera", cam_options)

# Image adjustments
gamma = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0, 0.1)
brightness = st.sidebar.slider("Brightness", -100, 100, 0, 1)

# Start/stop
run = st.sidebar.toggle("Start Camera")

frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(camera_index)
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        faces = list(faces_frontal) + list(faces_profile)

        for (x, y, w, h) in faces:
            x, y = max(x - 10, 0), max(y - 10, 0)
            w, h = min(w + 20, frame.shape[1] - x), min(h + 20, frame.shape[0] - y)

            face_roi = frame[y:y+h, x:x+w]
            face_roi = adjust_gamma(face_roi, gamma)
            face_roi = adjust_contrast_brightness(face_roi, contrast, brightness)
            frame[y:y+h, x:x+w] = face_roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(Image.fromarray(frame_rgb), channels="RGB")

    cap.release()