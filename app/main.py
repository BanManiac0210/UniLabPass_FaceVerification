from fastapi import FastAPI
from app.api.endpoints import router
import uvicorn
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Giới hạn lượng RAM TensorFlow sử dụng
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)]
    )

app = FastAPI()

# Đăng ký API router
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Mặc định là 8000 nếu không có PORT
    print(f"Server chạy trên cổng {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
