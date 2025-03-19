from fastapi import APIRouter, UploadFile, File
from app.services.face_verification import compare_faces

router = APIRouter()

@router.get("/healthcheck")
async def healthcheck():
    return {"status": "Server is running"}

@router.post("/verify_faces")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    API nhận 2 ảnh khuôn mặt dưới dạng file, xử lý trực tiếp mà không lưu vào backend.
    """
    try:
        # Đọc file ảnh vào bộ nhớ
        image1_bytes = await file1.read()
        image2_bytes = await file2.read()

        # So sánh hai ảnh khuôn mặt
        result = compare_faces(image1_bytes, image2_bytes)

        return result
    except Exception as e:
        # Xử lý lỗi nếu có
        return {"error": str(e)}
