import os
import json
import re
import warnings
from typing import List, Dict
import logging

# Отключаем лишние логи
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pandas as pd
from datetime import datetime


# ============================================================
# ВЫЧИСЛЕНИЕ METRIC CER И WER
# ============================================================

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Вычисляет Character Error Rate (CER)"""
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()

    if not ref:
        return 0.0 if not hyp else 1.0

    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n] / m


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Вычисляет Word Error Rate (WER)"""
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[m][n] / m


# ============================================================
# МЕТОДЫ ПРЕДОБРАБОТКИ
# ============================================================

class ImagePreprocessor:
    """Класс с различными методами предобработки изображений"""

    @staticmethod
    def no_preprocessing(image: np.ndarray) -> Image.Image:
        """Без предобработки"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 5) -> Image.Image:
        """Медианный фильтр"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filtered = cv2.medianBlur(rgb, kernel_size)
        return Image.fromarray(filtered)

    @staticmethod
    def otsu_threshold(image: np.ndarray) -> Image.Image:
        """Метод Оцу (тёмный текст на светлом фоне)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> Image.Image:
        """Адаптивная бинаризация"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def morphological_processing(image: np.ndarray) -> Image.Image:
        """Морфологическая обработка"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def sharpening(image: np.ndarray) -> Image.Image:
        """Увеличение резкости"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(rgb, -1, kernel)
        return Image.fromarray(sharpened)

    @staticmethod
    def otsu_morphological(image: np.ndarray) -> Image.Image:
        """Оцу + морфологическая обработка"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def combined_method(image: np.ndarray) -> Image.Image:
        """Комбинированный метод: Gaussian blur + Otsu + Sauvola"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Sauvola (локальная бинаризация)
        mean = cv2.boxFilter(blurred, cv2.CV_32F, (25, 25))
        sqmean = cv2.boxFilter(blurred.astype(np.float32) ** 2, cv2.CV_32F, (25, 25))
        std = np.sqrt(np.maximum(sqmean - mean ** 2, 0))

        k = 0.2
        r = 128
        threshold = mean * (1 + k * ((std / r) - 1))
        sauvola = (blurred > threshold).astype(np.uint8) * 255

        # Комбинируем Otsu и Sauvola
        combined = cv2.bitwise_and(otsu, sauvola)

        rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def clahe_otsu(image: np.ndarray) -> Image.Image:
        """CLAHE + Otsu (улучшенный контраст)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)


# ============================================================
# ЗАГРУЗКА МОДЕЛИ TrOCR
# ============================================================

class TrOCRRecognizer:
    """Класс для распознавания текста с помощью TrOCR"""

    def __init__(self, use_gpu: bool = True):
        self.model_name = "cyrillic-trocr/trocr-handwritten-cyrillic"
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'

        print(f"🔧 Инициализация TrOCR на устройстве: {self.device}")
        print(f"📚 Загрузка модели: {self.model_name}")

        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print("✅ Модель успешно загружена!")

    def recognize_image(self, pil_image: Image.Image) -> str:
        """Распознает текст на одном изображении"""
        try:
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                )

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"⚠️ Ошибка распознавания: {e}")
            return ""


# ============================================================
# СБОР ДАННЫХ ИЗ ДАТАСЕТА
# ============================================================

def load_dataset(json_path: str) -> List[Dict]:
    """Загружает датасет из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    base_dir = os.path.dirname(json_path)

    for item in data:
        img_path = os.path.join(base_dir, item['image_name'])

        if not os.path.exists(img_path):
            print(f"⚠️ Файл не найден: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Ошибка чтения: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        for region in item.get('regions', []):
            gt_text = region.get('text', '').strip()
            if not gt_text:
                continue

            bbox = region['bbox']
            x, y, w, h = map(int, bbox)

            # Вырезаем область с небольшим отступом
            x1 = max(0, x - w // 2)
            y1 = max(0, y - h // 2)
            x2 = min(img_w, x + w // 2)
            y2 = min(img_h, y + h // 2)

            crop = img[y1:y2, x1:x2]

            samples.append({
                'image_name': item['image_name'],
                'field_name': region.get('field', 'unknown'),
                'gt_text': gt_text,
                'crop': crop,
                'bbox': bbox
            })

    print(f"📊 Загружено {len(samples)} образцов")
    return samples


# ============================================================
# ЗАПУСК ЭКСПЕРИМЕНТОВ
# ============================================================

def run_experiments(json_path: str, use_gpu: bool = True):
    """Запускает эксперименты со всеми методами предобработки"""

    # Определяем методы предобработки
    preprocessing_methods = {
        #'Без предобработки': ImagePreprocessor.no_preprocessing,
        'Медианный фильтр': ImagePreprocessor.median_filter,
        'Метод Оцу': ImagePreprocessor.otsu_threshold,
        'Адаптивная бинаризация': ImagePreprocessor.adaptive_threshold,
        'Морфологическая обработка': ImagePreprocessor.morphological_processing,
        'Увеличение резкости': ImagePreprocessor.sharpening,
        'Оцу + Морф. обработка': ImagePreprocessor.otsu_morphological,
        'Комбинированный (Gaussian + Otsu + Sauvola)': ImagePreprocessor.combined_method,
        'CLAHE + Otsu': ImagePreprocessor.clahe_otsu  # Добавил для сравнения
    }

    # Загружаем данные
    print("\n" + "=" * 80)
    print("ЗАГРУЗКА ДАТАСЕТА")
    print("=" * 80)
    samples = load_dataset(json_path)

    if not samples:
        print("❌ Нет данных для обработки!")
        return

    # Инициализируем распознаватель
    recognizer = TrOCRRecognizer(use_gpu=use_gpu)

    # Результаты экспериментов
    all_results = {}

    # Получаем уникальные поля
    field_names = list(set([s['field_name'] for s in samples]))
    field_names.sort()

    print("\n" + "=" * 80)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print(f"📊 Полей для анализа: {len(field_names)}")
    print(f"🔬 Методов предобработки: {len(preprocessing_methods)}")
    print(f"📝 Всего образцов: {len(samples)}")
    print("=" * 80)

    # Для каждого метода предобработки
    for method_name, preprocess_func in tqdm(preprocessing_methods.items(), desc="Методы предобработки"):
        print(f"\n🔬 Тестирование: {method_name}")

        results_by_field = {}
        all_cers = []

        # Для каждого образца
        for sample in tqdm(samples, desc=f"  Распознавание", leave=False):
            try:
                # Применяем предобработку
                processed_image = preprocess_func(sample['crop'])

                # Распознаем
                predicted_text = recognizer.recognize_image(processed_image)
                ALLOWED_CHARS_PATTERN = re.compile(r'[^а-яА-Я0-9NVX,!?.:;()`\'"\s-]')
                filtered = ALLOWED_CHARS_PATTERN.sub(' ', predicted_text)

                # Удаляем лишние пробелы
                filtered = re.sub(r'\s+', ' ', filtered)
                filtered = filtered.strip()
                # Вычисляем CER
                cer = calculate_cer(sample['gt_text'], filtered)
                all_cers.append(cer)

                # Сохраняем по полям
                field = sample['field_name']
                if field not in results_by_field:
                    results_by_field[field] = []
                results_by_field[field].append(cer)

            except Exception as e:
                print(f"    Ошибка: {e}")
                continue

        # Вычисляем средний CER по каждому полю
        field_avg_cer = {}
        for field, cers in results_by_field.items():
            field_avg_cer[field] = np.mean(cers) if cers else 0.0

        # Сохраняем результаты
        all_results[method_name] = {
            'avg_cer': np.mean(all_cers) if all_cers else 0.0,
            'fields': field_avg_cer
        }

        print(f"  ✅ Средний CER: {all_results[method_name]['avg_cer']:.4f}")

    # ============================================================
    # СОЗДАНИЕ ТАБЛИЦЫ РЕЗУЛЬТАТОВ
    # ============================================================

    print("\n" + "=" * 80)
    print("ФОРМИРОВАНИЕ ТАБЛИЦЫ РЕЗУЛЬТАТОВ")
    print("=" * 80)

    # Создаем DataFrame для таблицы
    table_data = []

    for method_name, results in all_results.items():
        row = {
            'Метод обработки': method_name,
            'Средний CER': f"{results['avg_cer']:.4f}"
        }

        # Добавляем CER по каждому полю
        for field in field_names:
            cer_value = results['fields'].get(field, 0.0)
            row[field] = f"{cer_value:.4f}"

        table_data.append(row)

    # Сортируем по среднему CER
    table_data.sort(key=lambda x: float(x['Средний CER']))

    # Создаем DataFrame
    df = pd.DataFrame(table_data)

    # Сохраняем в Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"experiment_results_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='CER_results', index=False)

        # Добавляем статистику
        stats_data = []
        for method_name, results in all_results.items():
            stats_data.append({
                'Метод обработки': method_name,
                'Средний CER': results['avg_cer'],
                'Качество': 'Отлично' if results['avg_cer'] < 0.1 else
                'Хорошо' if results['avg_cer'] < 0.3 else
                'Средне' if results['avg_cer'] < 0.5 else
                'Плохо'
            })
        df_stats = pd.DataFrame(stats_data)
        df_stats = df_stats.sort_values('Средний CER')
        df_stats.to_excel(writer, sheet_name='Статистика', index=False)

    # Сохраняем в CSV
    csv_path = f"experiment_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Сохраняем в JSON
    json_results = {}
    for method_name, results in all_results.items():
        json_results[method_name] = {
            'avg_cer': results['avg_cer'],
            'fields': results['fields']
        }

    json_path_output = f"experiment_results_{timestamp}.json"
    with open(json_path_output, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    # ============================================================
    # ВЫВОД ТАБЛИЦЫ
    # ============================================================

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print("\n📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА (CER):\n")

    # Форматированный вывод
    print(f"{'Метод обработки':<40} {'Средний CER':<12}", end='')
    for field in field_names[:5]:  # Показываем первые 5 полей для краткости
        print(f"{field:<15}", end='')
    print()
    print("-" * 100)

    for row in table_data:
        print(f"{row['Метод обработки']:<40} {row['Средний CER']:<12}", end='')
        for field in field_names[:5]:
            print(f"{row.get(field, '0.0000'):<15}", end='')
        print()

    if len(field_names) > 5:
        print(f"\n... и еще {len(field_names) - 5} полей")

    print("\n" + "=" * 80)
    print("ЛУЧШИЕ МЕТОДЫ ПО СРЕДНЕМУ CER:")
    print("=" * 80)

    for i, row in enumerate(table_data[:5], 1):
        print(f"{i}. {row['Метод обработки']}: {row['Средний CER']}")

    print(f"\n💾 Результаты сохранены:")
    print(f"   📁 Excel: {excel_path}")
    print(f"   📁 CSV: {csv_path}")
    print(f"   📁 JSON: {json_path_output}")

    return all_results, df


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    # Путь к вашему JSON файлу
    JSON_FILE = "../recognition_dataset/dataset.json"

    print("=" * 80)
    print("ЭКСПЕРИМЕНТЫ ПО СРАВНЕНИЮ МЕТОДОВ ПРЕДОБРАБОТКИ")
    print("=" * 80)
    print("\nБудут протестированы следующие методы:")
    print("  • Без предобработки")
    print("  • Медианный фильтр")
    print("  • Метод Оцу")
    print("  • Адаптивная бинаризация")
    print("  • Морфологическая обработка")
    print("  • Увеличение резкости")
    print("  • Оцу + Морф. обработка")
    print("  • Комбинированный (Gaussian + Otsu + Sauvola)")
    print("  • CLAHE + Otsu")
    print("=" * 80)

    # Запуск экспериментов
    results, df = run_experiments(
        json_path=JSON_FILE,
        use_gpu=True  # Использовать GPU если доступен
    )

    print("\n✅ Эксперименты завершены!")