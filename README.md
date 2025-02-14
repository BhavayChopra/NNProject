# PyroScope AI - Fire & Smoke Detection on Edge Devices

## Overview

PyroScope AI is an advanced fire and smoke detection system built using **YOLOv8**. Designed for real-time inference on edge devices, it ensures rapid response in fire-prone environments like forests, warehouses, and industrial sites. The model is optimized using **ONNX** for efficient deployment on low-power hardware.

---

## Features

- **Real-time detection**: Identifies fire and smoke using a webcam.
- **Optimized for edge devices**: Runs efficiently on NVIDIA Jetson, Raspberry Pi, and similar hardware.
- **ONNX acceleration**: Faster inference with CUDA and CPU support.
- **User-friendly interface**: Displays live detection results with bounding boxes and labels.

---

## Installation

### Requirements

- Python 3.8+
- OpenCV
- PyTorch
- ONNX & ONNX Runtime
- Ultralytics YOLO

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/pyroscope-ai.git
   cd pyroscope-ai
   Install dependencies:

bash
Copy
pip install -r requirements.txt
Convert YOLOv8 model to ONNX (if not done already):

bash
Copy
python pyroscope_edge.py
Usage
Run the detection script:

bash
Copy
python pyroscope_edge.py
Press q to exit the live detection.

Deployment on Edge Devices
Ensure ONNX Runtime is installed with CUDA support if using Jetson.

Optimize the ONNX model using TensorRT for better performance.

Use a USB or CSI camera for real-time video feed.

Contributing
Contributions are welcome! Feel free to:

Fork the repository

Submit issues

Open pull requests
