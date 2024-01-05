# [1] https://viblo.asia/p/opencv-with-python-part-2-L4x5xRRBZBM
# [2] https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale
# [3] https://ihoctot.com/cascadeclassifier-la-gi
# [4] https://www.geeksforgeeks.org/python-opencv-cv2-flip-method/
# [5] https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/?ref=lbp
# [6] https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
# [7] https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/?ref=lbp
# [8] https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/?ref=lbp
# [9] https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/?ref=lbp
import cv2
import os

cap = cv2.VideoCapture(0)  # [1] "0" là thiết bị đầu tiên (camera)
cap.set(3, 640)  # [1][2] width
cap.set(4, 480)  # [1][2] height
# [1] cap.get (propId, value)
# propId: 0 đến 18
# Mỗi số biểu thị một thuộc tính của video (Bộ định danh thuộc tính
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
# [3] Load a classifier from a file
face_id = input('\nAdd your ID: ');
print('\nFace updating, please see into the camera....')

count = 1

while (True):
    ret, img = cap.read()  # [1] return True/False. True -> khung được đọc đúng
    
    img = cv2.flip(img, 1)  # [4] lật ảnh 2D
    # 0 lật dọc theo trục x
    # 1 lật ngang theo trục y
    # -1 lật cả dọc và ngang
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [5] convert không gian màu
    faces = face_detector.detectMultiScale(gray, 1.3, 5)  # [6] to find faces
    # scaleFactor: 1.3 là giảm 30% kích thước, gia tăng cơ hội tìm dc ảnh phù hợp
    # minNeighbors: 5 là số lượng hàng xóm mà mỗi hình chữ nhật ứng viên phải có để giữ lại nó
    # giá trị cao hơn dẫn đến ít phát hiện hơn nhưng chất lượng cao hơn
    # [6] Nếu khuôn mặt được tìm thấy, trả về vị trí của các khuôn mặt được phát hiện dưới dạng Rect (x, y, w, h)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2);
        # [7] rectangle vẽ hình chữ nhật trên bất kỳ hình ảnh
        # rectangle(image, start_point, end_point, color BGR, thickness (px))
        count += 1
        cv2.imwrite("dataset/user." + str(face_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w]);
        # [8] imwrite lưu hình ảnh vào thiết bị vd dataset/user.3.150.jpg
        # imwrite(filename, image)
        cv2.imshow("image", img)
        # [9] imshow để hiển thị hình ảnh trong cửa sổ
    k = cv2.waitKey(100) & 0xff
    # đợi người dùng nhấn bàn phím
    # 0xFF là kiểu số Hexa cho 8 bits, tương tự như từ 0-255 hệ thập phân
    if k == 27:  # 27 là phím ESC
        break
        
print("\nPress ESC to exit...")
cap.release()  # [1] giải phóng webcam
cv2.destroyAllWindows()  # [1] đóng tất cả các cửa sổ đang mở
