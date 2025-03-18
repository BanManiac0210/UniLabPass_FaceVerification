from fastapi import FastAPI
from app.api.endpoints import router
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Đăng ký API router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
