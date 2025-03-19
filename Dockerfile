# Sử dụng Python 3.10 (hoặc phiên bản bạn cần)
FROM python:3.9

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ code vào container
COPY . /app

# Cài đặt các thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng cho ứng dụng
EXPOSE 8000

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
