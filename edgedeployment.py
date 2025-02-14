import torch
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Convert YOLOv8 model to ONNX (Only run once to generate the ONNX model)
def convert_to_onnx(model_path='pyroscope_ai.pt', onnx_path='pyroscope_ai.onnx'):
    model = YOLO(model_path)
    model.export(format='onnx', opset=11)  # Export to ONNX format
    print(f"Model converted and saved at {onnx_path}")

# Load the ONNX model
onnx_path = 'pyroscope_ai.onnx'
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Preprocessing function
def preprocess(image, input_size=(640, 640)):
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Run inference on edge device
def detect_fire():
    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_tensor = preprocess(frame)
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, ort_inputs)[0]
        
        # Process outputs (post-processing needed based on model format)
        for detection in outputs:
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf > 0.5:  # Confidence threshold
                label = "Fire" if int(cls) == 0 else "Smoke"
                color = (0, 0, 255) if label == "Fire" else (255, 165, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("PyroScope AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Uncomment to convert model first
# convert_to_onnx()

detect_fire()
