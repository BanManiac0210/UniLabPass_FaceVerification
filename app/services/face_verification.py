import numpy as np
from keras_facenet import FaceNet
from .face_extraction import extract_face

# Khởi tạo model
embedder = FaceNet()

def compare_faces(img1, img2):
    """
    So sánh hai ảnh khuôn mặt, trả về độ tương đồng.
    """
    face1 = extract_face(img1)
    face2 = extract_face(img2)

    if face1 is None or face2 is None:
        return {"match": False, "distance": -1}

    # Trích xuất vector đặc trưng
    face1_embedding = embedder.embeddings([face1])[0]
    face2_embedding = embedder.embeddings([face2])[0]

    # Tính khoảng cách cosine
    distance = np.linalg.norm(face1_embedding - face2_embedding)
    threshold = 0.6  # Ngưỡng để quyết định có khớp hay không

    match = distance < threshold  # match là kiểu numpy.bool_
    
    # Chuyển numpy.bool_ thành bool trước khi trả về
    match = bool(match)  # Chuyển đổi sang kiểu bool
    
    # Chuyển đổi numpy.float32 thành float
    distance = float(distance)

    return {"match": match, "distance": round(distance, 2)}
