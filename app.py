from flask import Flask, request, jsonify
import requests
from PIL import Image
import io

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return """
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Detect">
    </form>
    """

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    
    image = request.files['image'].read()
    
    # Send the image to the YOLO model service
    response = requests.post('http://yolo-model:8080/predict', files={'image': image})
    
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
