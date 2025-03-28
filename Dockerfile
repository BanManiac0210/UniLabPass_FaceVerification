# Sử dụng Python 3.9
FROM python:3.9.2

# Cài đặt thư viện cần thiết
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ project vào container
COPY . .

# Mở cổng 8000 để chạy FastAPI
EXPOSE 8000

# Chạy FastAPI bằng Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
