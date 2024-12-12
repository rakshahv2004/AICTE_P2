import cv2
import os
import numpy as np
from PIL import Image

# Ensure OpenCV contrib package is installed
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The folder '{path}' does not exist.")

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'jpeg', 'png'))]
    faceSamples = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Convert to grayscale
            imageNp = np.array(pilImage, 'uint8')
            
            # Extract ID from the filename
            try:
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
            except (IndexError, ValueError):
                print(f"Skipping invalid file name: {imagePath}")
                continue
            
            # Detect faces
            faces = detector.detectMultiScale(imageNp)
            if len(faces) == 0:
                print(f"No faces detected in {imagePath}. Skipping.")
                continue
            
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
    
    return faceSamples, Ids

try:
    path = 'TrainingImage'
    faces, Ids = getImagesAndLabels(path)

    if len(faces) == 0 or len(Ids) == 0:
        raise ValueError("No valid faces or IDs found. Ensure the images are correctly formatted and contain faces.")

    # Train the recognizer and save the model
    recognizer.train(faces, np.array(Ids))
    save_path = 'TrainingImageLabel/trainner.yml'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    recognizer.save(save_path)
    print("Model trained and saved successfully!")
except Exception as e:
    print(f"Error during training: {e}")