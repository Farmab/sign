import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# Load ASL alphabet icons for text-to-sign translation
ICON_PATH = "asl_icons"  # Folder where A-Z icons are stored as A.png, B.png, etc.

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Simulated function to recognize gesture (Placeholder)
def recognize_letter(landmarks):
    # Placeholder: return a dummy letter for now
    return "A"

# Webcam handler
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    result_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                letter = recognize_letter(landmarks)
                result_text += letter

        stframe.image(frame, channels='BGR')

        if st.button("Stop Camera"):
            break

    cap.release()
    return result_text

# Text-to-sign display
def show_sign_language(text):
    for char in text.upper():
        if char.isalpha():
            icon_path = os.path.join(ICON_PATH, f"{char}.png")
            if os.path.exists(icon_path):
                image = Image.open(icon_path)
                st.image(image, width=100, caption=char)

# Streamlit UI
st.title("🤟 Sign Language Translator WebApp")

tabs = st.tabs(["Sign to English", "Text to Sign"])

with tabs[0]:
    st.subheader("🔍 Recognize Sign Language from Camera")
    if st.button("Start Camera"):
        result = capture_and_predict()
        st.success(f"Detected: {result}")

with tabs[1]:
    st.subheader("💬 Translate Text to Sign Icons")
    user_input = st.text_input("Type your sentence here:")
    if user_input:
        show_sign_language(user_input)
