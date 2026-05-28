import os
from typing import List, Dict
from ultralytics import YOLO


class YOLODetector:
    """Класс для детекции объектов с помощью YOLO"""

    def __init__(self, model_path: str = "experiments/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Загрузка YOLO модели"""
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO модель загружена из {self.model_path}")
        else:
            print(f"⚠️ Модель не найдена: {self.model_path}")
            self.model = None

    def detect(self, image_path: str) -> List[Dict]:
        """Детекция bounding boxes с помощью YOLO"""
        if self.model is None:
            # Возвращаем тестовые данные для демонстрации
            return [
                {"bbox": [150, 80, 100, 30], "text": "", "confidence": 0.95},
                {"bbox": [260, 82, 90, 28], "text": "", "confidence": 0.93},
                {"bbox": [150, 150, 120, 30], "text": "", "confidence": 0.92},
            ]

        results = self.model(image_path)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    x = int(x1 + w // 2)
                    y = int(y1 + h // 2)
                    confidence = float(box.conf[0])

                    detections.append({
                        "bbox": [x, y, w, h],
                        "confidence": confidence,
                        "text": ""
                    })

        return detections