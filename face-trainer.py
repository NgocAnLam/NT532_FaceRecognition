# [1] https://pyimagesearch.com/2021/05/03/face-recognition-with-local-binary-patterns-lbps-and-opencv/ -> LBP

import cv2
import numpy as np
import os
from PIL import Image
img_dir = "dataset"  # thư mục
# sử dụng Local Binary Patterns Histograms
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
# hàm lấy ảnh và nhãn
def getImageAndLabels(img_dir):
    # os.listdir: Trả về một danh sách chứa các tên các file trong thư mục
    # os.path.join(img_dir, f): Nối đường dẫn vd dataset/user.3.150.jgp
    imagePaths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # biến đổi màu (L = Trắng đen)
        img_numpy = np.array(PIL_img, 'uint8')
        # chuyển hình ảnh thành mảng
        # uint8: số nguyên có dấu 8 bit
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # căt path qua "/" và căt path qua "." và lấy arr[1] => 3
        faces = detector.detectMultiScale(img_numpy)  # to find faces
        # Nếu khuôn mặt được tìm thấy, trả về vị trí của các khuôn mặt được phát hiện dưới dạng Rect (x, y, w, h)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])  # thêm ảnh vào mảng
            ids.append(id)  # thêm id vào mảng
    return faceSamples, ids
print("\nFace training. please wait...")
faces, ids = getImageAndLabels(img_dir)
recognizer.train(faces, np.array(ids))
recognizer.write("trainer/trainer.yml")  # Save
print("\n{0} faces are learned.".format(len(np.unique(ids))))  # In ra số người
