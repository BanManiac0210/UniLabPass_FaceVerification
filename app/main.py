from fastapi import FastAPI
from app.api.endpoints import router
import uvicorn
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# # Giới hạn số lượng threads mà TensorFlow sử dụng
# tf.config.threading.set_intra_op_parallelism_threads(4)  # Số lượng threads tối đa cho các phép toán song song
# tf.config.threading.set_inter_op_parallelism_threads(2)  # Số lượng threads tối đa cho các phép toán giữa các phép toán

# Giới hạn bộ nhớ sử dụng cho TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.set_visible_devices(gpu, 'GPU')  # Sử dụng GPU nếu có
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=256)])  # Giới hạn bộ nhớ cho mỗi GPU

app = FastAPI()

# Đăng ký API router
app.include_router(router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Mặc định là 8000 nếu không có PORT
    print(f"Server chạy trên cổng {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
