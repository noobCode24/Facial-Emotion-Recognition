import cv2
from keras.models import model_from_json
import numpy as np
import os
import time
from tkinter import Tk, filedialog
from collections import Counter
import matplotlib.pyplot as plt

def load_model():
    """Tải mô hình CNN từ file JSON và trọng số."""
    json_file = open("emotion_model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("emotion_model.h5")
    return model

# Tải Haar Cascade để phát hiện khuôn mặt
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Định nghĩa nhãn cảm xúc
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Tạo thư mục screenshots nếu chưa có
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = np.array(face).reshape(1, 48, 48, 1)
    return face / 255.0

def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)
    else:
        ratio = width / float(w)
        height = int(h * ratio)
    return cv2.resize(image, (width, height))

def detect_emotion_from_image(model, detector, labels):
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("Chưa chọn ảnh.")
        return

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotion_data = []  # Danh sách lưu trữ cảm xúc phát hiện được để thống kê

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        processed_face = preprocess_face(face)

        pred = model.predict(processed_face)
        emotion_idx = np.argmax(pred)
        emotion_label = labels[emotion_idx]
        confidence = pred[0][emotion_idx]

        if confidence > 0.4:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            emotion_data.append(emotion_label)  # Thêm cảm xúc phát hiện vào danh sách

    resized_image = resize_image(image, width=800)
    cv2.imshow("Emotion Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh vào thư mục screenshots
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join("screenshots", f"{timestamp}_emotion_detected.png")
    cv2.imwrite(screenshot_path, image)
    print(f"Ảnh đã được lưu tại {screenshot_path}")

    # Tính toán và hiển thị thống kê cảm xúc
    if emotion_data:
        emotion_counts = Counter(emotion_data)
        total_emotions = sum(emotion_counts.values())
        emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()}
        print("Phân bố cảm xúc:")
        for emotion, percentage in emotion_percentages.items():
            print(f"{emotion}: {percentage:.2f}%")
        # Vẽ biểu đồ bánh cho phân bố cảm xúc
        plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(), autopct="%1.1f%%", startangle=90)
        plt.title("Phân bố cảm xúc")
        plt.show()
    else:
        print("Không phát hiện được cảm xúc trong ảnh.")

def detect_emotion_from_webcam(model, detector, labels):
    emotion_data = []
    start_time = time.time()

    webcam = cv2.VideoCapture(0)
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Không thể truy cập webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            processed_face = preprocess_face(face)

            pred = model.predict(processed_face)
            emotion_idx = np.argmax(pred)
            emotion_label = labels[emotion_idx]
            confidence = pred[0][emotion_idx]

            if confidence > 0.4:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                elapsed_time = time.time() - start_time
                emotion_data.append((emotion_label, elapsed_time))
                screenshot_path = os.path.join("screenshots", f"{elapsed_time:.2f}_{emotion_label}.png")
                cv2.imwrite(screenshot_path, frame)

        resized_frame = resize_image(frame, width=800)
        cv2.imshow("Emotion Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

    # Tính toán và hiển thị thống kê cảm xúc cho webcam
    if emotion_data:
        emotion_counts = Counter([item[0] for item in emotion_data])
        total_emotions = sum(emotion_counts.values())
        emotion_percentages = {emotion: (count / total_emotions) * 100 for emotion, count in emotion_counts.items()}
        print("Phân bố cảm xúc:")
        for emotion, percentage in emotion_percentages.items():
            print(f"{emotion}: {percentage:.2f}%")
        # Vẽ biểu đồ bánh cho phân bố cảm xúc
        plt.pie(emotion_percentages.values(), labels=emotion_percentages.keys(), autopct="%1.1f%%", startangle=90)
        plt.title("Phân bố cảm xúc")
        plt.show()
    else:
        print("Không phát hiện được cảm xúc trong phiên làm việc.")

def main():
    model = load_model()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    while True:
        print("\n=== Chọn chế độ ===")
        print("1. WebCam")
        print("2. Choice Image")
        print("3. Quit")

        choice = input("Nhập lựa chọn của bạn (1/2/3): ")

        if choice == '1':
            detect_emotion_from_webcam(model, detector, labels)
        elif choice == '2':
            detect_emotion_from_image(model, detector, labels)
        elif choice == '3':
            confirm = input("Bạn có chắc muốn thoát? (y/n): ")
            if confirm.lower() == 'y':
                print("Thoát chương trình.")
                break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBạn muốn dừng chương trình? (y/n): ", end="")
        if input().lower() == 'y':
            print("Chương trình đã dừng.")
        else:
            main()
