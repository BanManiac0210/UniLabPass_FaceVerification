from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from app.services.face_detection import detect_face
from app.services.face_verification import verify_faces

router = APIRouter()

def read_image(file: UploadFile):
    """Chuyển ảnh từ UploadFile sang numpy array."""
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@router.get("/healthcheck")
async def healthcheck():
    """API kiểm tra trạng thái hoạt động của server."""
    return {"code": 200, "message": "Server is running"}

@router.post("/verify")
async def face_verification(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    """API xác minh khuôn mặt."""
    img1 = read_image(image1)
    img2 = read_image(image2)

    # Phát hiện khuôn mặt
    faces1 = detect_face(img1)
    faces2 = detect_face(img2)
    
    if (len(faces1) == 0 and len(faces2) == 0):
        return {"error": "No face detected in both images"}
    if (len(faces1) == 0):
        return {"error": "No face detected in image 1"}
    if (len(faces2) == 0):
        return {"error": "No face detected in image 2"}

    # Cắt khuôn mặt đầu tiên phát hiện được
    x1, y1, x2, y2 = faces1[0]
    face1 = img1[y1:y2, x1:x2]

    x1, y1, x2, y2 = faces2[0]
    face2 = img2[y1:y2, x1:x2]

    # Xác minh khuôn mặt
    result = verify_faces(face1, face2)
    return result
