import cv2
import numpy as np
from PIL import Image
from typing import List

class ImagePreprocessor:
    """Класс для предобработки изображений"""

    @staticmethod
    def preprocess_for_trocr(image: np.ndarray) -> Image.Image:
        """Предобработка изображения для TrOCR"""
        # Конвертация в RGB если нужно
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.medianBlur(image, 5)
        # Применение CLAHE для улучшения контраста
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(image)

    @staticmethod
    def crop_bbox(image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """Вырезает область по bounding box с отступом"""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]

        x1 = max(0, x - w // 2 - padding)
        y1 = max(0, y - h // 2 - padding)
        x2 = min(img_w, x + w // 2 + padding)
        y2 = min(img_h, y + h // 2 + padding)

        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]

            # Поворачиваем если высота больше ширины
            if h > w * 1.2:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            return crop

        return None