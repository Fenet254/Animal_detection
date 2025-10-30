from ultralytics import YOLO
import torch
import os

class AnimalDetector:
    def __init__(self, model_path='yolov5s.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_animals(self, image):
        """Detect animals in the image and return results."""
        results = self.model(image, conf=0.5, classes=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24])  # COCO classes for animals: cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
        return results

    def get_detected_animals(self, results):
        """Extract detected animal names and confidences."""
        animals = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                label = result.names[cls]
                animals.append({'name': label, 'confidence': conf})
        return animals

    def train_model(self, data_path, epochs=10, batch_size=16, img_size=640):
        """Train the YOLO model on custom dataset."""
        self.model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size, device=self.device)
        # Save the trained model
        self.model.save('trained_model.pt')
        return 'trained_model.pt'
