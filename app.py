import streamlit as st
import face_recognition
import numpy as np
import cv2
import os
from PIL import Image

st.title("Face Recognition App")

KNOWN_FACES_DIR = "C:\\Users\\harsh\\Desktop\\firstproject\\known_faces"
TOLERANCE = 0.5

known_faces = []
known_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)

    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

st.write("Known Faces Loaded:", known_names)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Draw box
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Processed Image")