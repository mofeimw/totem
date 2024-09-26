import os
from flask import Flask, request, jsonify, send_from_directory
from openvino.inference_engine import IECore
import cv2
import numpy as np
from transformers import pipeline

app = Flask(__name__)

# Initialize OpenVINO
ie = IECore()
net = ie.read_network(model="yolov3.xml", weights="yolov3.bin")
exec_net = ie.load_network(network=net, device_name="CPU")

# Initialize transformer for text generation
generator = pipeline('text-generation', model='distilgpt2')

# Load class names
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def preprocess_image(image, input_shape):
    resized = cv2.resize(image, input_shape)
    resized = resized.transpose((2, 0, 1))  # HWC to CHW
    return resized[np.newaxis, ...]  # Add batch dimension

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detect_and_generate', methods=['POST'])
def detect_and_generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read and preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    input_blob = next(iter(exec_net.inputs))
    input_shape = exec_net.inputs[input_blob].shape[2:]
    preprocessed_img = preprocess_image(img, input_shape)

    # Perform inference
    outputs = exec_net.infer(inputs={input_blob: preprocessed_img})
    output_blob = next(iter(exec_net.outputs))
    detections = outputs[output_blob][0][0]

    # Process detections
    detected_objects = []
    for detection in detections:
        confidence = float(detection[2])
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detection[1])
            detected_objects.append(class_names[class_id])

    # Generate recipe using transformer
    prompt = f"Generate a recipe using these ingredients: {', '.join(detected_objects)}"
    recipe = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

    return jsonify({'detected_objects': detected_objects, 'recipe': recipe})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
