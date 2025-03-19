import numpy as np
from keras_facenet import FaceNet
from .face_extraction import extract_face

# Khởi tạo model
embedder = FaceNet()

def compare_faces(img1, img2):
    # Lấy khuôn mặt đã chuẩn hóa từ ảnh
    face1 = extract_face(img1)
    face2 = extract_face(img2)

    # Trích xuất vector đặc trưng
    embeddings = embedder.embeddings([face1, face2])
    face1_embedding = embeddings[0]
    face2_embedding = embeddings[1]

    # Tính khoảng cách cosine giữa hai ảnh
    distance = np.linalg.norm(face1_embedding - face2_embedding)
    threshold = 0.6
    match = distance < threshold

    return {"match": bool(match), "distance": round(distance, 2)}
