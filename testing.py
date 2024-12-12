import cv2
import os
import numpy as np

# Ensure OpenCV contrib package is installed
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the trained model
model_path = 'TrainingImageLabel/trainner.yml'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at {model_path}. Ensure training was successful.")

recognizer.read(model_path)

# Initialize the webcam
camera = cv2.VideoCapture(0)  # 0 for default webcam, or replace with video file path
font = cv2.FONT_HERSHEY_SIMPLEX

print("Press 'q' to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        Id, confidence = recognizer.predict(face)  # Predict the ID

        # Display the results
        if confidence < 100:
            name = f"ID: {Id}, Confidence: {round(100 - confidence, 2)}%"
        else:
            name = "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(Id), (x, y - 10), font, 2, (255, 255, 255), 3)

    # Show the frame
    cv2.imshow('Face Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()