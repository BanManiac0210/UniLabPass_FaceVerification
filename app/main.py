from fastapi import FastAPI
from app.api.endpoints import router
import uvicorn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Đăng ký API router
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Mặc định là 8000 nếu không có PORT
    print(f"Server chạy trên cổng {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
