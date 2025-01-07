import cv2 #Thư viện OpenCV (Open Source Computer Vision Library) được sử dụng để xử lý hình ảnh và video.
# Đọc và xử lý hình ảnh từ tệp hoặc webcam.
# Phát hiện khuôn mặt bằng cách sử dụng Haar Cascade (một thuật toán học máy).
# Hiển thị hình ảnh với các hình chữ nhật bao quanh khuôn mặt và nhãn cảm xúc.
# Chuyển đổi hình ảnh từ định dạng màu BGR (Blue, Green, Red) sang đen trắng (gray) để xử lý.
from keras.models import model_from_json 
# Keras là một thư viện để xây dựng và huấn luyện các mô hình học sâu. Hàm model_from_json được sử dụng để tải mô hình CNN
import numpy as np
#Numpy là thư viện chủ yếu dùng để xử lý mảng (array) và thực hiện các phép toán ma trận.
import os
#  Thư viện này cung cấp các chức năng tương tác với hệ điều hành, như làm việc với tệp và thư mục
import time
from tkinter import Tk, filedialog
#  Đây là thư viện tiêu chuẩn của Python dùng để xây dựng giao diện người dùng đồ họa (GUI).
from collections import Counter
# Counter là một lớp trong thư viện collections giúp đếm số lần xuất hiện của các phần tử trong một danh sách hoặc một đối tượng lặp.
import matplotlib.pyplot as plt
#  Matplotlib là một thư viện vẽ đồ họa trong Python.


def load_model():
    """Tải mô hình CNN từ file JSON và trọng số."""
    json_file = open("emotion_model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json) #xay dung mo hinh dua tren chuoi json
    model.load_weights("emotion_model.h5") #gan cac trong so da huyan luyen vao mo hinh da xay dung truoc do 
    return model

# Tạo thư mục screenshots nếu chưa có
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# thực hiện việc tiền xử lý một ảnh khuôn mặt trước khi đưa nó vào mô hình học sâu để dự đoán cảm xúc.
def preprocess_face(face):
    face = cv2.resize(face, (48, 48)) #Thay đổi kích thước ảnh khuôn mặt đầu vào (face) về kích thước cố định là 48x48 pixel.
    face = np.array(face).reshape(1, 48, 48, 1) 
    #Chuyển đổi ảnh thành mảng NumPy để dễ dàng xử lý và tương thích với mô hình.
    #Thay đổi hình dạng của mảng NumPy để phù hợp với đầu vào của mô hình
    return face / 255.0 #Chuẩn hóa giá trị pixel của ảnh từ 0-255 về phạm vi 0-1.

# Hàm resize_image được thiết kế để thay đổi kích thước ảnh một cách tỷ lệ,
# đảm bảo giữ nguyên tỉ lệ khung hình (aspect ratio) ban đầu của ảnh
def resize_image(image, width=None, height=None):
    (h, w) = image.shape[:2] #Lấy chiều cao (h) và chiều rộng (w) của ảnh từ thuộc tính .shape.
    if width is None and height is None:
        return image # Nếu cả width và height đều không được cung cấp, tức là không yêu cầu thay đổi kích thước.
    if width is None:
        ratio = height / float(h) #Tính tỷ lệ thay đổi dựa trên chiều rộng mới (width) so với chiều rộng gốc (w).
        width = int(w * ratio)
    else:
        ratio = width / float(w)
        height = int(h * ratio)
    return cv2.resize(image, (width, height))

def detect_emotion_from_image(model, detector, labels):
    Tk().withdraw() #Ẩn giao diện chính của Tkinter.
    #Hiển thị cửa sổ chọn file để người dùng chọn ảnh từ hệ thống.
    #Chỉ cho phép chọn các file định dạng .jpg, .jpeg, và .png.
    image_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("Chưa chọn ảnh.")
        return

    # Đọc ảnh từ đường dẫn đã chọn.
    image = cv2.imread(image_path)
    # Chuyển ảnh từ màu (BGR) sang ảnh xám để giảm dữ liệu cần xử lý.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Dùng mô hình Haar Cascade để phát hiện các khuôn mặt.
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotion_data = []  # Danh sách lưu trữ cảm xúc phát hiện được để thống kê

    for (x, y, w, h) in faces:
        #Cắt phần khuôn mặt từ ảnh xám.
        face = gray[y:y + h, x:x + w]
        #Chuẩn hóa ảnh khuôn mặt (kích thước 48x48, chuẩn hóa giá trị pixel).
        processed_face = preprocess_face(face)
        # Sử dụng mô hình CNN để dự đoán cảm xúc.
        pred = model.predict(processed_face)
        #Lấy chỉ số của cảm xúc có xác suất cao nhất.
        emotion_idx = np.argmax(pred)
        #Tra tên cảm xúc từ danh sách labels.
        emotion_label = labels[emotion_idx]
        #Xác suất dự đoán của cảm xúc.
        confidence = pred[0][emotion_idx]

        if confidence > 0.4:
            #Vẽ khung chữ nhật xung quanh khuôn mặt.
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #Hiển thị nhãn cảm xúc cùng xác suất.
            cv2.putText(image, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            emotion_data.append(emotion_label)  # Thêm cảm xúc phát hiện vào danh sách

    # Đảm bảo ảnh hiển thị vừa với màn hình, không quá lớn.
    resized_image = resize_image(image, width=800)
    #Hiển thị ảnh với cảm xúc được phát hiện.
    cv2.imshow("Emotion Detection", resized_image)
    #Dừng màn hình chờ người dùng đóng cửa sổ.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Lưu ảnh vào thư mục screenshots
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join("screenshots", f"{timestamp}_emotion_detected.png")
    cv2.imwrite(screenshot_path, image)
    print(f"Ảnh đã được lưu tại {screenshot_path}")

    # Tính toán và hiển thị thống kê cảm xúc
    if emotion_data:
        #Đếm số lần xuất hiện của mỗi cảm xúc.
        emotion_counts = Counter(emotion_data)
        total_emotions = sum(emotion_counts.values())
        #Tính phần trăm của mỗi cảm xúc.
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
    # Danh sách lưu trữ cảm xúc phát hiện được để thống kê
    emotion_data = []
    #Ghi lại thời gian bắt đầu phiên làm việc để tính thời điểm phát hiện từng cảm xúc.
    start_time = time.time()

    # Mở kết nối với webcam. 0 chỉ định webcam mặc định trên máy.
    webcam = cv2.VideoCapture(0)
    while True:
        #Đọc một khung hình từ webcam.
        ret, frame = webcam.read()
        if not ret:
            print("Không thể truy cập webcam.")
            break
        
         # Chuyển ảnh từ màu (BGR) sang ảnh xám để giảm dữ liệu cần xử lý.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # phat hien khuon mat
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            #Cắt phần khuôn mặt từ ảnh xám.
            face = gray[y:y + h, x:x + w]
            #Chuẩn hóa ảnh khuôn mặt (kích thước 48x48, chuẩn hóa giá trị pixel).
            processed_face = preprocess_face(face)
            
            # Sử dụng mô hình CNN để dự đoán cảm xúc.
            pred = model.predict(processed_face)
             #Lấy chỉ số của cảm xúc có xác suất cao nhất.
            emotion_idx = np.argmax(pred)
            #Tra tên cảm xúc từ danh sách labels.
            emotion_label = labels[emotion_idx]
            #Xác suất dự đoán của cảm xúc.
            confidence = pred[0][emotion_idx]

            if confidence > 0.4:
                #ve khung chữ nhật xung quanh khuôn mặt.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #Hiển thị nhãn cảm xúc cùng xác suất.
                cv2.putText(frame, f"{emotion_label} ({confidence * 100:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                # Ghi lại thời gian và cảm xúc vào emotion_data.
                elapsed_time = time.time() - start_time
                emotion_data.append((emotion_label, elapsed_time))
                # Lưu khung hình hiện tại vào thư mục screenshots với tên chứa thời gian và nhãn cảm xúc.
                screenshot_path = os.path.join("screenshots", f"{elapsed_time:.2f}_{emotion_label}.png")
                cv2.imwrite(screenshot_path, frame)

        # Điều chỉnh kích thước khung hình để phù hợp với màn hình
        resized_frame = resize_image(frame, width=800)
        # Hiển thị ảnh với cảm xúc được phát hiện.
        cv2.imshow("Emotion Detection", resized_frame)
        #Đợi phím bấm, dừng nếu người dùng nhấn q.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên của webcam.
    webcam.release()
    # Đóng tất cả cửa sổ hiển thị.
    cv2.destroyAllWindows()

    # Tính toán và hiển thị thống kê cảm xúc cho webcam
    if emotion_data:
         #Đếm số lần xuất hiện của mỗi cảm xúc.
        emotion_counts = Counter([item[0] for item in emotion_data])
        total_emotions = sum(emotion_counts.values())
        # Tính tỷ lệ phần trăm của từng cảm xúc.
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

    # Tải Haar Cascade để phát hiện khuôn mặt
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Định nghĩa nhãn cảm xúc
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
