import streamlit as st
import numpy as np
import cv2
import pickle
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from io import BytesIO
from googleapiclient.http import MediaIoBaseDownload
from sklearn.svm import SVC

# Load the face detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Google Drive Setup (optional)
def upload_to_drive(file_path, file_name):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = service_account.Credentials.from_service_account_info(st.secrets["gcp"])
    drive_service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_name, 'parents': ['17kt0LrNDI5IZpvwMML-S6r_kjGy5jLgR']}
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Function for face data capture
def capture_face_data(name):
    progress_placeholder = st.empty()  # Placeholder for updating progress
    face_data = []
    st.write("Processing...")
    placeholder = st.empty()

    skip = 0

    # Check for stop condition
    if st.button("Stop capturing facial data", key="btn4"):
        return face_data  # Early return if capturing is stopped

    while True:
        frame = st.camera_input("Capture Face Data")  # Use Streamlit's camera input
        if frame is not None:
            frame = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for face in faces[:1]:  # Only the largest face
                x, y, w, h = face
                offset = 5
                face_offset = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                face_selection = cv2.resize(face_offset, (100, 100))

                if skip % 10 == 0:
                    face_data.append(face_selection)
                    st.write(f"Processing: {len(face_data)}%")

                skip += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(face_data) >= 100:
                break

            placeholder.image(frame, channels="BGR")

    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    np.save(f'./face_dataset/{name}.npy', face_data)

    # Optionally, upload to Google Drive
    file_id = upload_to_drive(f'./face_dataset/{name}.npy', f'{name}.npy')
    st.write(f"Data saved to Google Drive with ID: {file_id}")

def face_rec():
    placeholder = st.empty()

    # User Input
    st.write("Enter the name for data capture:")
    name = st.text_input("Name")

    if st.button("Capture Face Data"):
        capture_face_data(name)

    # Real-time face recognition
    if st.button("Run Face Recognition"):
        st.write("Starting face recognition...")

        # Assuming you have implemented the list_files_in_folder and load_face_data functions properly
        folder_id = '17kt0LrNDI5IZpvwMML-S6r_kjGy5jLgR'  # Replace with your actual folder ID
        items, drive_service = list_files_in_folder(folder_id)

        # Load face data from the listed files
        face_data, labels, names = load_face_data(drive_service, items)

        # Train the model on the loaded face data
        clf = train_model(face_data, labels)

        # Recognition loop
        while True:
            frame = st.camera_input("Run Face Recognition")  # Use Streamlit's camera input
            if frame is not None:
                frame = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_section = frame[y - 5:y + h + 5, x - 5:x + w + 5]
                    face_section = cv2.resize(face_section, (100, 100)).flatten().reshape(1, -1)

                    prediction = clf.predict(face_section)[0]
                    user = names[int(prediction)]
                    st.write(f"Recognized: {user}")

                    cv2.putText(frame, user, (x + 20, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                placeholder.image(frame, channels="BGR")

# Main App
def main():
    st.title("Face Recognition App")
    face_rec()

if __name__ == "__main__":
    main()
