import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import time

# Load model
json_file = open("emotion_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotion_model.h5")

# Load Haar Cascade
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels for emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create folder for screenshots if not exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Helper function to preprocess face
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = np.array(face).reshape(1, 48, 48, 1)
    return face / 255.0

# Initialize variables for emotion tracking
emotion_data = []
start_time = time.time()

# Start video capture
webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        processed_face = preprocess_face(face)

        # Predict emotion
        pred = model.predict(processed_face)
        emotion_idx = np.argmax(pred)
        emotion_label = labels[emotion_idx]
        confidence = pred[0][emotion_idx]

        # Draw rectangle and put emotion label
        if confidence > 0.4:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Record detected emotion
            elapsed_time = time.time() - start_time
            emotion_data.append((emotion_label, elapsed_time))

            # Save screenshot
            screenshot_path = os.path.join("screenshots", f"{elapsed_time:.2f}_{emotion_label}.png")
            cv2.imwrite(screenshot_path, frame)

    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

# Analyze and visualize results
if emotion_data:
    emotion_counts = Counter([item[0] for item in emotion_data])
    total_emotions = sum(emotion_counts.values())

    # Calculate percentages
    emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()}

    # Print statistics
    print("Emotion Distribution:")
    for emotion, percentage in emotion_percentages.items():
        print(f"{emotion}: {percentage:.2f}%")

    # Plot pie chart
    plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(), autopct="%1.1f%%", startangle=90)
    plt.title("Emotion Distribution")
    plt.show()
else:
    print("No emotions detected during the session.")
