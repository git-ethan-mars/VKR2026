import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2


class TrOCRRecognizer:
    """Класс для распознавания текста с помощью TrOCR"""

    def __init__(self, model_name: str = "cyrillic-trocr/trocr-handwritten-cyrillic"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = None
        self.load_model()

    def load_model(self):
        """Загрузка TrOCR модели"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Используется устройство: {self.device}")

        print("📚 Загрузка TrOCR модели...")
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ TrOCR модель загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки TrOCR: {e}")
            # Fallback на английскую модель
            fallback_model = "microsoft/trocr-base-printed"
            self.processor = TrOCRProcessor.from_pretrained(fallback_model)
            self.model = VisionEncoderDecoderModel.from_pretrained(fallback_model)
            self.model.to(self.device)
            self.model.eval()

    def recognize(self, image: np.ndarray, preprocessor=None) -> str:
        """Распознавание текста на изображении"""
        try:
            # Предобработка изображения
            if preprocessor:
                pil_image = preprocessor(image)
            else:
                # Базовая предобработка
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)

            # Подготовка для модели
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
    
            # Генерация текста
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )

            # Декодирование
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text

        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return ""