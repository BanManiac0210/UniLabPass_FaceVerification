import cv2
import numpy as np

# Load model phát hiện khuôn mặt
face_net = cv2.dnn.readNetFromCaffe(
    "app/models/deploy.prototxt", 
    "app/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

def detect_face(image):
    """Phát hiện khuôn mặt trong ảnh."""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Ngưỡng nhận diện khuôn mặt
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))

    return faces
