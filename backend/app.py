from flask import Flask, request, jsonify
from ultralytics import YOLO
from transformers import pipeline
import cv2
import numpy as np
from openvino.runtime import Core

app = Flask(__name__)

# Initialize YOLO model
# yolo_model = YOLO('yolov8n.pt')

ie = Core()
yolo_model = ie.read_model("path/to/yolo_openvino_model.xml")
compiled_model = ie.compile_model(yolo_model, "CPU")

# Initialize LLM
llm = pipeline("text-generation", model="gpt2")

@app.route('/detect_and_generate', methods=['POST'])
def detect_and_generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read and process the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Detect food items
    results = compiled(img)
    detected_foods = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
    
    # Generate recipe using LLM
    prompt = f"Generate a recipe using these ingredients: {', '.join(detected_foods)}"
    recipe = llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    
    return jsonify({'detected_foods': detected_foods, 'recipe': recipe})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
