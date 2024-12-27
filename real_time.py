import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import time

# cv2: Sử dụng OpenCV để truy cập webcam, phát hiện khuôn mặt, và xử lý hình ảnh thời gian thực.
# keras.models.model_from_json: Tải cấu trúc mô hình CNN từ tệp JSON.
# numpy: Xử lý mảng số học, chuẩn hóa ảnh đầu vào.
# matplotlib.pyplot: Trực quan hóa kết quả, như vẽ biểu đồ phân phối cảm xúc.
# collections.Counter: Đếm số lần các cảm xúc được phát hiện.
# os: Quản lý tệp/thư mục.
# time: Theo dõi thời gian.


# Tải mô hình CNN đã được huấn luyện.
json_file = open("emotion_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotion_model.h5")

# Tải Haar Cascade để phát hiện khuôn mặt
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Định nghĩa nhãn cảm xúc
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Tạo thư mục lưu ảnh (nếu chưa có)
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Hàm tiền xử lý khuôn mặt
# resize: Chuyển kích thước ảnh về 48x48 pixel (phù hợp với đầu vào của mô hình).
# reshape: Định dạng lại ảnh thành (1, 48, 48, 1) (batch size, height, width, channels).
# / 255.0: Chuẩn hóa giá trị pixel về khoảng [0, 1].
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = np.array(face).reshape(1, 48, 48, 1)
    return face / 255.0

# Khởi tạo biến theo dõi
# emotion_data: Lưu danh sách các cảm xúc đã được phát hiện và thời gian tương ứng.
# start_time: Ghi lại thời gian bắt đầu phiên làm việc.
emotion_data = []
start_time = time.time()

# Kích hoạt webcam để nhận luồng video.
webcam = cv2.VideoCapture(0)
while True:
    #  Lấy một khung hình từ webcam.
    # ret: Biến boolean, xác định xem khung hình có được đọc thành công không.
    # frame: Khung hình hiện tại.
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Chuyển đổi ảnh sang thang xám và phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # gray[y:y + h, x:x + w]: Cắt khuôn mặt từ khung hình.
    # preprocess_face: Chuẩn bị khuôn mặt cho mô hình.
    # model.predict: Dự đoán xác suất cho từng cảm xúc.
    # np.argmax(pred): Lấy chỉ số cảm xúc có xác suất cao nhất.
    # confidence: Độ tin cậy (xác suất) của cảm xúc được dự đoán.
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        processed_face = preprocess_face(face)

        # Predict emotion
        pred = model.predict(processed_face)
        emotion_idx = np.argmax(pred)
        emotion_label = labels[emotion_idx]
        confidence = pred[0][emotion_idx]

        # Draw rectangle and put emotion label
        # Vẽ khung hình chữ nhật quanh khuôn mặt.
        # Hiển thị cảm xúc và độ tin cậy lên khung hình.
        # Lưu ảnh vào thư mục screenshots nếu độ tin cậy > 0.4.
        if confidence > 0.4:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Record detected emotion
            elapsed_time = time.time() - start_time
            emotion_data.append((emotion_label, elapsed_time))

            # Save screenshot
            screenshot_path = os.path.join("screenshots", f"{elapsed_time:.2f}_{emotion_label}.png")
            cv2.imwrite(screenshot_path, frame)

    # Hiển thị khung hình hiện tại.
    # Thoát khỏi chương trình nếu nhấn phím q.
    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng webcam và giải phóng tài nguyên
webcam.release()
cv2.destroyAllWindows()

# Analyze and visualize results
if emotion_data:
    # Counter: Đếm số lần từng cảm xúc xuất hiện.
    # emotion_percentages: Tính tỷ lệ phần trăm cho từng cảm xúc. 
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
