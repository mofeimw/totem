from flask import Flask, request, jsonify
from openshift_ai import ObjectDetection
import numpy as np
import cv2

app = Flask(__name__)

# Initialize the YOLO object detection model
model = ObjectDetection()

@app.route('/')
def index():
    return "Welcome to OpenShift AI YOLO Object Detection"

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    detections = model.detect(image)

    # Process and return results
    results = []
    for detection in detections:
        results.append({
            'class': detection.class_name,
            'confidence': float(detection.confidence),
            'bbox': detection.bbox.tolist()
        })

    return jsonify({'detections': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
