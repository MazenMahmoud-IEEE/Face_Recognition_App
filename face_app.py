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
    creds = service_account.Credentials.from_service_account_file(
        r".\face-recognition-app-440518-e50c6b2c8d3a.json", scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_name, 'parents': ['17kt0LrNDI5IZpvwMML-S6r_kjGy5jLgR']}
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Function for face data capture
def capture_face_data(name, cap):
    progress_placeholder = st.empty()  # Placeholder for updating progress
    face_data = []
    st.write("Processing...")
    placeholder = st.empty()

    skip = 0
    # Check for stop condition
    if st.button("Stop capturing facial data",key="btn4"):
        cap.release()
        cv2.destroyAllWindows()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

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

        if len(face_data) >= 150:
            break

        placeholder.image(frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    np.save(f'./face_dataset/{name}.npy', face_data)

    # Optionally, upload to Google Drive
    file_id = upload_to_drive(f'./face_dataset/{name}.npy', f'{name}.npy')
    st.write(f"Data saved to Google Drive with ID: {file_id}")

# Google Drive File Listing
def list_files_in_folder(folder_id):
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(
        r".\face-recognition-app-440518-e50c6b2c8d3a.json", scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=creds)

    query = f"'{folder_id}' in parents"
    
    try:
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        return results.get('files', []), drive_service
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return []

def load_face_data(drive_service, items):
    face_data = []
    labels = []
    names = {}
    class_id = 0

    for item in items:
        file_name = item['name']
        
        if file_name.endswith('.npy'):
            # Store name mapping
            names[class_id] = file_name[:-4]

            # Download the .npy file
            request = drive_service.files().get_media(fileId=item['id'])
            file_stream = BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                st.write(f'Download {int(status.progress() * 100)}%.')

            # Load face data
            file_stream.seek(0)  # Move to the beginning of the BytesIO stream
            data_item = np.load(file_stream, allow_pickle=True)
            face_data.append(data_item)

            # Create labels
            target = class_id * np.ones((data_item.shape[0],))
            labels.append(target)
            class_id += 1

    return face_data, labels, names

# Function to load the trained model from a file
@st.cache_resource
def load_model():
    model_path = r".\face_recognition_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
            return clf
    return None

# Function to save the trained model to a file
def save_model(clf):
    model_path = r".\face_recognition_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

# Function to train the model
def train_model(face_data, labels):
    all_face_data = np.concatenate(face_data, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    # Train the model
    clf = SVC()  # Using SVM as an example
    clf.fit(all_face_data, all_labels)
    return clf

def face_rec():
    placeholder = st.empty()

    # Load the existing model
    clf = load_model()
    if clf is not None:
        st.write("Loaded existing model.")

    if st.button("Capture Face Data"):
        cap = cv2.VideoCapture(0)
        capture_face_data(name, cap)

    # Real-time face recognition
    if st.button("Run Face Recognition"):
        cap = cv2.VideoCapture(0)
        st.write("Starting face recognition...")

        # List files in the Google Drive folder
        folder_id = '17kt0LrNDI5IZpvwMML-S6r_kjGy5jLgR'  # Replace with your actual folder ID
        items, drive_service = list_files_in_folder(folder_id)

        # Load face data from the listed files
        face_data, labels, names = load_face_data(drive_service, items)

        # Train the model on the loaded face data
        if clf is None:  # Train new model if no existing model
            clf = train_model(face_data, labels)
        else:  # Update existing model with new data
            new_face_data = np.concatenate(face_data, axis=0)
            new_labels = np.concatenate(labels, axis=0)
            clf.fit(new_face_data, new_labels)

        # Save the updated model
        save_model(clf)

        # Check for stop condition
        if st.button("Stop face recognition",key="btn4"):
            cap.release()
            cv2.destroyAllWindows()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

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
        
    # User Input
    st.write("Enter the name for data capture:")
    name = st.text_input("Name")

# Main App
def main():
    st.title("Face Recognition App")
    face_rec()

if __name__ == "__main__":
    main()
