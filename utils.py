import cv2
import numpy as np
from PIL import Image
import zipfile
import os
import shutil

def load_image(image_file):
    """Load image from file or bytes."""
    if isinstance(image_file, str):
        image = cv2.imread(image_file)
    else:
        image = Image.open(image_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def preprocess_image(image, max_size=640):
    """Resize image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

def draw_boxes(image, results):
    """Draw bounding boxes and labels on image."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = result.names[cls]
            if conf > 0.5 and label in ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:  # Filter for animals
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def extract_zip(zip_path, extract_to='data'):
    """Extract zip file to specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def prepare_dataset(data_dir):
    """Prepare dataset by organizing images and labels."""
    # Assuming the dataset is in YOLO format with images and labels folders
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    return data_dir
