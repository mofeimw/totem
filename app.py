from flask import Flask, request, jsonify
import yolov5
import cv2
import numpy as np

app = Flask(__name__)

# Load the YOLOv5 model
model = yolov5.load('yolov5s')

@app.route('/')
def index():
    return "Welcome to YOLOv5 Object Detection"

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    results = model(image)

    # Process and return results
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
