from openshift_ai import ObjectDetection as OAIObjectDetection

class ObjectDetection:
    def __init__(self):
        self.model = OAIObjectDetection.from_pretrained("yolov5")

    def detect(self, image):
        return self.model(image)
