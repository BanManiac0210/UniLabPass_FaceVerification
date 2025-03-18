import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def extract_face(image_bytes):
    """
    Trích xuất khuôn mặt từ ảnh (PNG, JPEG, WebP,...) được gửi dưới dạng byte stream.
    Trả về khuôn mặt đã chuẩn hóa dưới dạng mảng numpy (160x160).
    """
    # Đọc ảnh từ byte stream bằng OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Không thể đọc ảnh từ byte stream.")

    # Phát hiện khuôn mặt trong ảnh
    faces = detector.detect_faces(img)

    if not faces:
        raise ValueError("Không phát hiện thấy khuôn mặt nào trong ảnh.")

    # Giả sử chúng ta chỉ lấy khuôn mặt đầu tiên được phát hiện
    face = faces[0]

    # Lấy các tọa độ của khuôn mặt (x, y, width, height)
    x, y, width, height = face['box']

    # Cắt phần khuôn mặt từ ảnh
    face_img = img[y:y+height, x:x+width]

    # Thay đổi kích thước khuôn mặt về 160x160 để phù hợp với FaceNet
    face_img = cv2.resize(face_img, (160, 160))

    return face_img
