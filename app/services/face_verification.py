import cv2
import numpy as np
import onnxruntime as ort

# Load model MobileFaceNet ONNX
onnx_model_path = "app/models/mobilefacenet.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

def preprocess_face(face):
    """Tiền xử lý khuôn mặt cho MobileFaceNet."""
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))  # Chuyển (H, W, C) -> (C, H, W)
    face = np.expand_dims(face, axis=0)  # Thêm batch dimension
    return face

def extract_feature(face):
    """Trích xuất đặc trưng khuôn mặt."""
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    face = preprocess_face(face)
    feature = ort_session.run([output_name], {input_name: face})[0]
    return feature

def cosine_similarity(feature1, feature2):
    """Tính độ tương đồng Cosine giữa hai vector đặc trưng."""
    dot_product = np.dot(feature1, feature2.T)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    return dot_product / (norm1 * norm2)

def verify_faces(face1, face2, threshold = 0.5):
    """Xác minh hai khuôn mặt có phải cùng một người không."""
    feature1 = extract_feature(face1)
    feature2 = extract_feature(face2)
    similarity = cosine_similarity(feature1, feature2)
    if (similarity > threshold):
        return {"code": 200, "samePerson": True, "similarity": float(similarity)}
    else:
        return {"code": 200, "samePerson": False, "similarity": float(similarity)}
    
