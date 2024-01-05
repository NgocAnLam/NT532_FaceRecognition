# [1] https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/?ref=lbp

import RPi.GPIO as GPIO  # thư viện dùng để điều khiển các chân GPIO của Raspberry Pi
import cv2
import numpy as np
import os
import pyttsx3  # Chuyển đổi văn bản thành giọng nói
import speech_recognition as sr  # Chuyển đổi giọng nói thành văn bản

r = sr.Recognizer()  # khởi tạo speech_recognition
robot_mouth = pyttsx3.init()  # khởi tạo pyttsx3
engine_brain = ""

recognizer = cv2.face.LBPHFaceRecognizer_create()  # khởi tạo LBPHFaceRecognizer
recognizer.read('trainer/trainer.yml')  # Tải KQ train được lưu trong trainer.yml
cascade = 'Cascades/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascade)  # Tải tệp Cascade để phát hiện khuôn mặt
font = cv2.FONT_HERSHEY_SIMPLEX

GPIO.setmode(GPIO.BCM)  # kiểu đánh số chân BCM trên Raspberry Pi
GPIO.setup(14, GPIO.OUT)  # Để thiết lập chân 14 là ngõ ra
id = 0
names = ['', 'Khuongtd','Thinhvp','Phongbp','Khoanha']  # Mảng lưu tên đối tưởng nhận dạng

cap = cv2.VideoCapture(0)  # "0" là thiết bị đầu tiên (camera)
cap.set(3, 640);  # width
cap.set(4, 480);  # height
# Xác định kích thước cửa sổ tối thiểu để được nhận dạng là một khuôn mặt
minW = 0.1*cap.get(3)  # 64
minH = 0.1*cap.get(4)  # 48

while True:
    ret, frame = cap.read()  # return True/False. True -> khung được đọc đúng
    img = cv2.flip(frame, 1)  # [4] lật ảnh 2D
    # 0 lật dọc theo trục x
    # 1 lật ngang theo trục y
    # -1 lật cả dọc và ngang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert không gian màu
    faces = faceCascade.detectMultiScale(  # to find faces
        gray,
        scaleFactor=1.2,  # scaleFactor: 1.2 là giảm 20% kích thước, gia tăng cơ hội tìm dc ảnh phù hợp
        minNeighbors=5,   # minNeighbors: 5 là số lượng hàng xóm mà mỗi hình chữ nhật ứng viên phải có để giữ lại nó
        minSize=(int(minW), int(minH))  # minSize: Kích thước đối tượng tối thiểu. Các đối tượng nhỏ hơn bị bỏ qua
    )
    # [6] Nếu khuôn mặt được tìm thấy, trả về vị trí của các khuôn mặt được phát hiện dưới dạng Rect (x, y, w, h)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # rectangle vẽ hình chữ nhật trên bất kỳ hình ảnh
        # rectangle(image, start_point, end_point, color BGR, thickness (px))
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        # predict: hàm dự đoán return id và độ chính xác
        print(str(id) + " => " + str(confidence))
        if (confidence < 80):  # Nếu nhận ra khuôn mặt và có trong danh sách
            id = names[id]
            confidence = " {0}".format(round(confidence))  # lm tròn độ chính xác
           
            #GPIO.output(14, 1)
        else:  # Nếu nhận khuôn mặt nhưng không có trong danh sách thì "unknown"
            id = "unknown"
            confidence = " {0}".format(round(confidence))  # lm tròn độ chính xác

        # [1] putText để viết văn bản trên hình ảnh
        # putText(image, text, org, font, fontScale, color (BGR), thickness)
        # org: tọa độ của góc dưới bên trái của chuỗi văn bản trong hình ảnh
        # frontScale: Hệ số tỷ lệ phông chữ được nhân với kích thước cơ bản của phông chữ cụ thể
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        
    cv2.imshow("Camera", img)  # hiển thị hình ảnh trong cửa sổ
    #GPIO.output(14, 0)

    # đợi người dùng nhấn bàn phím
    # 0xFF là kiểu số Hexa cho 8 bits, tương tự như từ 0-255 hệ thập phân
    k = cv2.waitKey(10) & 0xff
    if k == 27:  # 27 là phím ESC
        break

print("\nPress ESC to exit...")
cap.release()  # giải phóng webcam
cv2.destroyAllWindows()  # đóng tất cả các cửa sổ đang mở
                    
    

