import os
import uuid
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import warnings

warnings.filterwarnings('ignore')

# Создаем необходимые директории
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="OCR Web App", description="Распознавание текста с помощью YOLO + TrOCR")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Глобальные переменные для моделей
yolo_model = None
trocr_processor = None
trocr_model = None
device = None

# Допустимые символы: русские буквы, цифры, знаки препинания и кавычки
# а-я, А-Я, 0-9, N, V, X, !, ?, ., ,, :, ;, ), (, `, ", ' (одинарные и двойные кавычки)
ALLOWED_CHARS_PATTERN = re.compile(r'[^а-яА-Я0-9NVX,!?.:;()`\'"\s-]')


def validate_and_filter_text(text: str) -> Optional[str]:
    """
    Валидация и фильтрация текста.
    Удаляет символы, не входящие в разрешенный алфавит.

    Разрешенные символы:
    - Русские буквы (а-я, А-Я)
    - Цифры (0-9)
    - Буквы N, V, X
    - Знаки препинания: ! ? . , : ; ( ) `
    - Кавычки: " и '
    - Пробелы и дефисы

    Args:
        text: Входной текст для валидации

    Returns:
        Отфильтрованный текст или None, если текст пуст после фильтрации
    """
    if not text or not text.strip():
        return None

    # Заменяем недопустимые символы на пробелы
    filtered = ALLOWED_CHARS_PATTERN.sub(' ', text)

    # Удаляем лишние пробелы
    filtered = re.sub(r'\s+', ' ', filtered)
    filtered = filtered.strip()

    # Проверяем, есть ли что-то кроме пробелов
    if not filtered or len(filtered.strip()) == 0:
        return None

    return filtered


def is_bbox_inside(inner_bbox: List[int], outer_bbox: List[int]) -> bool:
    """
    Проверяет, содержится ли один bounding box полностью внутри другого.

    Args:
        inner_bbox: Внутренний bbox [x, y, w, h] (центр, ширина, высота)
        outer_bbox: Внешний bbox [x, y, w, h] (центр, ширина, высота)

    Returns:
        True если inner_bbox полностью внутри outer_bbox
    """
    # Преобразуем в координаты углов
    inner_x1 = inner_bbox[0] - inner_bbox[2] // 2
    inner_y1 = inner_bbox[1] - inner_bbox[3] // 2
    inner_x2 = inner_bbox[0] + inner_bbox[2] // 2
    inner_y2 = inner_bbox[1] + inner_bbox[3] // 2

    outer_x1 = outer_bbox[0] - outer_bbox[2] // 2
    outer_y1 = outer_bbox[1] - outer_bbox[3] // 2
    outer_x2 = outer_bbox[0] + outer_bbox[2] // 2
    outer_y2 = outer_bbox[1] + outer_bbox[3] // 2

    # Проверяем, находится ли внутренний bbox полностью внутри внешнего
    return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and
            inner_x2 <= outer_x2 and inner_y2 <= outer_y2)


def remove_nested_bboxes(detections: List[Dict]) -> List[Dict]:
    """
    Удаляет bounding boxes, которые полностью содержатся внутри других.

    Алгоритм:
    1. Сортирует блоки по размеру (от большего к меньшему)
    2. Для каждого блока проверяет, не содержится ли он внутри какого-либо большего блока
    3. Удаляет вложенные блоки

    Args:
        detections: Список детекций

    Returns:
        Отфильтрованный список без вложенных блоков
    """
    if len(detections) <= 1:
        return detections

    # Создаем копию списка
    detections_copy = detections.copy()

    # Сортируем по площади (от большего к меньшему)
    detections_copy.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)

    to_remove = set()

    for i, outer_det in enumerate(detections_copy):
        if i in to_remove:
            continue

        outer_bbox = outer_det["bbox"]

        for j, inner_det in enumerate(detections_copy):
            if i == j or j in to_remove:
                continue

            inner_bbox = inner_det["bbox"]

            # Проверяем, содержится ли inner внутри outer
            if is_bbox_inside(inner_bbox, outer_bbox):
                to_remove.add(j)
                print(f"   🗑️ Удален вложенный блок: inner={inner_bbox} внутри outer={outer_bbox}")

    # Фильтруем список, оставляя только не вложенные блоки
    filtered_detections = [det for idx, det in enumerate(detections_copy) if idx not in to_remove]

    # Восстанавливаем исходный порядок
    filtered_detections.sort(key=lambda d: d.get("original_index", 0))

    return filtered_detections


def filter_empty_detections(detections: List[Dict]) -> List[Dict]:
    """
    Фильтрует детекции, у которых после валидации текст стал пустым.

    Args:
        detections: Список детекций

    Returns:
        Отфильтрованный список детекций
    """
    filtered_detections = []

    for detection in detections:
        text = detection.get("text", "")
        validated_text = validate_and_filter_text(text)

        if validated_text is not None:
            detection["text"] = validated_text
            filtered_detections.append(detection)
        else:
            print(f"   🗑️ Удален пустой блок после валидации")

    return filtered_detections


def validate_batch_texts(detections: List[Dict]) -> List[Dict]:
    """
    Валидация текстов для всех детекций.

    Args:
        detections: Список детекций

    Returns:
        Список детекций с валидированными текстами
    """
    for detection in detections:
        text = detection.get("text", "")
        validated_text = validate_and_filter_text(text)

        if validated_text is not None:
            detection["text"] = validated_text
            detection["was_validated"] = True
        else:
            detection["text"] = ""
            detection["was_validated"] = False
            detection["is_empty"] = True

    return detections


def load_models():
    """Загрузка моделей YOLO и TrOCR"""
    global yolo_model, trocr_processor, trocr_model, device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Используется устройство: {device}")

    # Загрузка YOLO модели
    model_path = "experiments/weights/best.pt"
    if os.path.exists(model_path):
        yolo_model = YOLO(model_path)
        print(f"✅ YOLO модель загружена из {model_path}")
    else:
        print(f"⚠️ Модель не найдена: {model_path}")
        yolo_model = None

    # Загрузка TrOCR модели
    print("📚 Загрузка TrOCR модели...")
    try:
        trocr_processor = TrOCRProcessor.from_pretrained("cyrillic-trocr/trocr-handwritten-cyrillic")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("cyrillic-trocr/trocr-handwritten-cyrillic")
        trocr_model.to(device)
        trocr_model.eval()
        print("✅ TrOCR модель загружена")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки TrOCR: {e}")
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        trocr_model.to(device)
        trocr_model.eval()


def preprocess_for_trocr(image: np.ndarray) -> Image.Image:
    """Предобработка изображения для TrOCR"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    return Image.fromarray(image)


def recognize_text_with_trocr(image: np.ndarray) -> str:
    """Распознавание текста с помощью TrOCR"""
    try:
        pil_image = preprocess_for_trocr(image)
        pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            generated_ids = trocr_model.generate(
                pixel_values,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                temperature=0.7
            )

        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Валидация распознанного текста
        validated_text = validate_and_filter_text(text)

        return validated_text if validated_text else ""
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        return ""


def merge_nearby_bboxes(detections: List[Dict], horizontal_threshold: float = 1.5, vertical_threshold: float = 0.5) -> \
List[Dict]:
    """
    Объединяет близко расположенные bounding boxes в строки.
    """
    if not detections:
        return detections

    detections = sort_bboxes_top_to_bottom(detections)

    merged_lines = []
    used = [False] * len(detections)

    for i, det in enumerate(detections):
        if used[i]:
            continue

        x, y, w, h = det["bbox"]
        line_boxes = [det]
        used[i] = True

        for j, other in enumerate(detections):
            if used[j]:
                continue

            ox, oy, ow, oh = other["bbox"]

            y_center_diff = abs((y - h / 2) - (oy - oh / 2))
            vertical_match = y_center_diff < vertical_threshold * max(h, oh)

            if vertical_match:
                x_distance = abs(x - ox)
                horizontal_match = x_distance < horizontal_threshold * max(w, ow)

                if horizontal_match:
                    line_boxes.append(other)
                    used[j] = True

        line_boxes.sort(key=lambda b: b["bbox"][0] - b["bbox"][2] / 2)

        if len(line_boxes) > 1:
            merged_bbox = merge_bboxes([b["bbox"] for b in line_boxes])
            merged_confidence = np.mean([b["confidence"] for b in line_boxes])

            merged_lines.append({
                "bbox": merged_bbox,
                "confidence": merged_confidence,
                "text": "",
                "original_boxes": line_boxes,
                "is_merged": True
            })
        else:
            line_boxes[0]["is_merged"] = False
            merged_lines.append(line_boxes[0])

    return merged_lines


def merge_bboxes(bboxes: List[List[int]]) -> List[int]:
    """Объединяет несколько bounding boxes в один."""
    corners = []
    for x, y, w, h in bboxes:
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        corners.append([x1, y1, x2, y2])

    x1 = min(c[0] for c in corners)
    y1 = min(c[1] for c in corners)
    x2 = max(c[2] for c in corners)
    y2 = max(c[3] for c in corners)

    w = x2 - x1
    h = y2 - y1
    x = x1 + w // 2
    y = y1 + h // 2

    return [x, y, w, h]


def sort_bboxes_top_to_bottom(detections: List[Dict]) -> List[Dict]:
    """Сортировка bounding boxes сверху вниз, слева направо."""
    if not detections:
        return detections

    avg_height = sum(d["bbox"][3] for d in detections) / len(detections)
    line_threshold = avg_height * 0.8

    detections_with_y = []
    for idx, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        top_y = y - h // 2
        detections_with_y.append({
            **det,
            "original_index": idx,
            "top_y": top_y,
            "center_x": x,
        })

    detections_with_y.sort(key=lambda d: d["top_y"])

    rows = []
    current_row = []
    prev_y = None

    for det in detections_with_y:
        if prev_y is None or abs(det["top_y"] - prev_y) <= line_threshold:
            current_row.append(det)
        else:
            if current_row:
                rows.append(current_row)
            current_row = [det]
        prev_y = det["top_y"]

    if current_row:
        rows.append(current_row)

    sorted_detections = []
    for row in rows:
        row.sort(key=lambda d: d["center_x"])
        for det in row:
            original_index = det.pop("original_index", 0)
            det.pop("top_y", None)
            det.pop("center_x", None)
            sorted_detections.append(det)

    return sorted_detections


def detect_bboxes_with_yolo(image_path: str) -> List[Dict]:
    """Детекция bounding boxes с помощью YOLO."""
    if yolo_model is None:
        # Тестовые данные
        return [
            {"bbox": [150, 80, 100, 30], "text": "", "confidence": 0.95},
            {"bbox": [260, 82, 90, 28], "text": "", "confidence": 0.93},
            {"bbox": [150, 150, 120, 30], "text": "", "confidence": 0.92},
        ]

    results = yolo_model(image_path)
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


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCR распознавание текста</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1600px;
                margin: 0 auto;
            }

            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }

            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }

            .upload-area {
                background: white;
                border-radius: 20px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .upload-area:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 50px rgba(0,0,0,0.15);
            }

            .upload-area.drag-over {
                background: #f0f0ff;
                border: 2px dashed #667eea;
            }

            .upload-icon {
                font-size: 48px;
                margin-bottom: 20px;
            }

            .upload-text {
                color: #666;
                font-size: 1.1rem;
            }

            .upload-input {
                display: none;
            }

            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1rem;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 20px;
                display: inline-block;
            }

            .btn:hover {
                transform: scale(1.05);
            }

            .main-content {
                display: flex;
                gap: 20px;
                min-height: 600px;
            }

            .image-section {
                flex: 2;
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }

            .image-section h2 {
                margin-bottom: 15px;
                color: #333;
            }

            .canvas-wrapper {
                position: relative;
                display: inline-block;
                width: 100%;
                text-align: center;
            }

            .image-canvas {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
                cursor: pointer;
            }

            .transcriptions-section {
                flex: 1;
                background: white;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                max-height: 80vh;
                overflow-y: auto;
            }

            .transcriptions-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                flex-shrink: 0;
            }

            .transcriptions-header h2 {
                color: #333;
                margin: 0;
            }

            .transcriptions-list {
                flex: 1;
                overflow-y: auto;
            }

            .transcription-item {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                transition: all 0.2s;
                cursor: pointer;
                border: 2px solid transparent;
            }

            .transcription-item:hover {
                background: #e9ecef;
                transform: translateX(5px);
            }

            .transcription-item.selected {
                border-color: #667eea;
                background: #f0f0ff;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
            }

            .transcription-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                color: #667eea;
                font-weight: bold;
            }

            .transcription-text {
                width: 100%;
                padding: 10px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1rem;
                font-family: inherit;
                resize: vertical;
            }

            .transcription-text:focus {
                outline: none;
                border-color: #667eea;
            }

            .confidence {
                font-size: 0.85rem;
                color: #28a745;
            }

            .merged-badge {
                font-size: 0.75rem;
                background: #667eea;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                margin-left: 10px;
            }

            .validated-badge {
                font-size: 0.75rem;
                background: #28a745;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                margin-left: 5px;
            }

            .save-btn {
                background: #28a745;
                margin-top: 0;
            }

            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }

            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .toast {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #333;
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                display: none;
                z-index: 1000;
                animation: slideIn 0.3s ease;
            }

            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            .info-text {
                text-align: center;
                color: #999;
                margin-top: 20px;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📝 OCR Распознавание текста</h1>
                <p>YOLOv11 + TrOCR (с валидацией символов и удалением вложенных блоков)</p>
            </div>

            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📸</div>
                <div class="upload-text">
                    <strong>Нажмите или перетащите изображение</strong><br>
                    Поддерживаются форматы: JPG, PNG, JPEG
                </div>
                <input type="file" id="fileInput" class="upload-input" accept="image/*">
                <button class="btn" onclick="document.getElementById('fileInput').click()">Выбрать файл</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">Обработка изображения... Это может занять несколько секунд</p>
            </div>

            <div class="main-content" id="mainContent" style="display: none;">
                <div class="image-section">
                    <h2>🖼️ Изображение с разметкой</h2>
                    <div class="canvas-wrapper">
                        <canvas id="imageCanvas" class="image-canvas"></canvas>
                    </div>
                    <div class="info-text">💡 Кликните на прямоугольник, чтобы выбрать блок для редактирования</div>
                </div>

                <div class="transcriptions-section">
                    <div class="transcriptions-header">
                        <h2>📋 Распознанный текст</h2>
                        <button class="btn save-btn" id="saveBtn" onclick="saveTranscriptions()">💾 Сохранить</button>
                    </div>
                    <div id="transcriptionsList" class="transcriptions-list"></div>
                </div>
            </div>
        </div>

        <div id="toast" class="toast"></div>

        <script>
            let currentImage = null;
            let currentDetections = [];
            let currentFilename = null;
            let canvas = null;
            let ctx = null;
            let scale = 1;
            let selectedIndex = -1;

            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const loading = document.getElementById('loading');
            const mainContent = document.getElementById('mainContent');
            const imageCanvas = document.getElementById('imageCanvas');
            canvas = imageCanvas;
            ctx = canvas.getContext('2d');

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleFile(file);
                }
            });

            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    handleFile(file);
                }
            });

            async function handleFile(file) {
                const formData = new FormData();
                formData.append('file', file);

                loading.style.display = 'block';
                mainContent.style.display = 'none';
                selectedIndex = -1;

                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    currentFilename = data.filename;
                    currentDetections = data.detections;

                    const img = new Image();
                    img.onload = () => {
                        drawImageWithBoxes(img, data.detections);
                        displayTranscriptions(data.detections);
                        mainContent.style.display = 'flex';
                        loading.style.display = 'none';
                    };
                    img.src = data.url;
                    currentImage = img;

                    const emptyCount = data.detections.filter(d => !d.text || d.text.trim() === '').length;
                    if (emptyCount > 0) {
                        showToast(`⚠️ ${emptyCount} блоков были удалены (пустой текст после валидации)`, 'warning');
                    } else {
                        showToast('✅ Изображение успешно обработано!', 'success');
                    }
                } catch (error) {
                    console.error('Ошибка:', error);
                    showToast('❌ Ошибка при обработке изображения', 'error');
                    loading.style.display = 'none';
                }
            }

            function drawImageWithBoxes(img, detections) {
                const maxWidth = 800;
                scale = Math.min(maxWidth / img.width, 1);
                const width = img.width * scale;
                const height = img.height * scale;

                canvas.width = width;
                canvas.height = height;

                ctx.drawImage(img, 0, 0, width, height);

                detections.forEach((detection, index) => {
                    const [x, y, w, h] = detection.bbox;
                    const scaledX = (x - w/2) * scale;
                    const scaledY = (y - h/2) * scale;
                    const scaledW = w * scale;
                    const scaledH = h * scale;

                    if (index === selectedIndex) {
                        ctx.strokeStyle = '#ff4444';
                        ctx.lineWidth = 3;
                    } else {
                        ctx.strokeStyle = detection.is_merged ? '#28a745' : '#667eea';
                        ctx.lineWidth = 2;
                    }
                    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

                    ctx.fillStyle = index === selectedIndex ? '#ff4444' : (detection.is_merged ? '#28a745' : '#667eea');
                    ctx.font = 'bold 12px Arial';
                    ctx.fillText(`#${index + 1}`, scaledX + 5, scaledY + 20);
                });
            }

            function displayTranscriptions(detections) {
                const container = document.getElementById('transcriptionsList');
                container.innerHTML = '';

                detections.forEach((detection, index) => {
                    const item = document.createElement('div');
                    item.className = 'transcription-item';
                    if (index === selectedIndex) {
                        item.classList.add('selected');
                    }
                    item.id = `transcription-${index}`;
                    item.onclick = () => selectTranscription(index);

                    const confidencePercent = detection.confidence ? (detection.confidence * 100).toFixed(1) : 'N/A';
                    const mergedBadge = detection.is_merged ? '<span class="merged-badge">📎 строка</span>' : '';
                    const validatedBadge = detection.was_validated ? '<span class="validated-badge">✓ валидирован</span>' : '';

                    item.innerHTML = `
                        <div class="transcription-header">
                            <span>📦 Блок #${index + 1} ${mergedBadge} ${validatedBadge}</span>
                            <span class="confidence">Уверенность: ${confidencePercent}%</span>
                        </div>
                        <textarea 
                            class="transcription-text" 
                            id="text-${index}"
                            data-original="${escapeHtml(detection.text)}"
                            rows="3"
                            onchange="updateText(${index})"
                        >${escapeHtml(detection.text)}</textarea>
                    `;

                    container.appendChild(item);
                });
            }

            function selectTranscription(index) {
                selectedIndex = index;

                document.querySelectorAll('.transcription-item').forEach((item, i) => {
                    if (i === index) {
                        item.classList.add('selected');
                    } else {
                        item.classList.remove('selected');
                    }
                });

                if (currentImage) {
                    drawImageWithBoxes(currentImage, currentDetections);
                }

                const selectedElement = document.getElementById(`transcription-${index}`);
                if (selectedElement) {
                    selectedElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }

            function updateText(index) {
                const textarea = document.getElementById(`text-${index}`);
                if (textarea) {
                    currentDetections[index].text = textarea.value;
                }
            }

            function escapeHtml(text) {
                if (!text) return '';
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            async function saveTranscriptions() {
                const transcriptions = [];

                currentDetections.forEach((detection, index) => {
                    const textarea = document.getElementById(`text-${index}`);
                    if (textarea) {
                        const currentText = textarea.value;
                        const originalText = textarea.dataset.original || '';

                        transcriptions.push({
                            bbox: detection.bbox,
                            text: currentText,
                            original: originalText,
                            confidence: detection.confidence,
                            edited: currentText !== originalText,
                            order: index + 1,
                            is_merged: detection.is_merged || false,
                            was_validated: detection.was_validated || false
                        });
                    }
                });

                const data = {
                    filename: currentFilename,
                    image_name: currentFilename,
                    transcriptions: transcriptions
                };

                try {
                    const response = await fetch('/save', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();
                    if (result.success) {
                        showToast(`✅ Транскрипции сохранены в файл: ${result.file}`, 'success');
                        setTimeout(() => {
                            window.open(`/download/${result.file}`, '_blank');
                        }, 1000);
                    }
                } catch (error) {
                    console.error('Ошибка сохранения:', error);
                    showToast('❌ Ошибка при сохранении', 'error');
                }
            }

            function showToast(message, type) {
                const toast = document.getElementById('toast');
                toast.textContent = message;
                toast.style.display = 'block';
                toast.style.backgroundColor = type === 'success' ? '#28a745' : (type === 'warning' ? '#ff9800' : '#dc3545');

                setTimeout(() => {
                    toast.style.display = 'none';
                }, 3000);
            }

            canvas.addEventListener('click', (e) => {
                if (!currentDetections.length) return;

                const rect = canvas.getBoundingClientRect();
                const clickX = (e.clientX - rect.left) * (canvas.width / rect.width);
                const clickY = (e.clientY - rect.top) * (canvas.height / rect.height);

                let closestIndex = -1;

                currentDetections.forEach((detection, index) => {
                    const [x, y, w, h] = detection.bbox;
                    const scaledX = (x - w/2) * scale;
                    const scaledY = (y - h/2) * scale;
                    const scaledW = w * scale;
                    const scaledH = h * scale;

                    if (clickX >= scaledX && clickX <= scaledX + scaledW &&
                        clickY >= scaledY && clickY <= scaledY + scaledH) {
                        closestIndex = index;
                    }
                });

                if (closestIndex !== -1) {
                    selectTranscription(closestIndex);
                }
            });

            console.log('Приложение загружено');
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Загрузка изображения и распознавание"""
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join("uploads", filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Получаем все детекции от YOLO
    raw_detections = detect_bboxes_with_yolo(file_path)

    print(f"\n📊 Обработка изображения: {filename}")
    print(f"   Найдено блоков до фильтрации: {len(raw_detections)}")

    # Удаляем вложенные bounding boxes
    detections = remove_nested_bboxes(raw_detections)
    print(f"   После удаления вложенных блоков: {len(detections)}")

    # Объединяем близкие блоки в строки
    detections = merge_nearby_bboxes(detections)
    print(f"   После объединения в строки: {len(detections)}")

    img = cv2.imread(file_path)
    img_h, img_w = img.shape[:2]

    valid_detections = []

    for idx, detection in enumerate(detections):
        x, y, w, h = detection["bbox"]
        expand_padding = 5

        x1 = max(0, x - w // 2 - expand_padding)
        y1 = max(0, y - h // 2 - expand_padding)
        x2 = min(img_w, x + w // 2 + expand_padding)
        y2 = min(img_h, y + h // 2 + expand_padding)

        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]

            if h > w * 1.2:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            text = recognize_text_with_trocr(crop)

            # Валидация текста
            validated_text = validate_and_filter_text(text)

            if validated_text:
                detection["text"] = validated_text
                detection["was_validated"] = True
                detection["order"] = len(valid_detections) + 1
                valid_detections.append(detection)

                if detection.get("is_merged"):
                    print(f"   ✅ Строка #{len(valid_detections)}: '{validated_text}' (объединено, валидировано)")
                else:
                    print(f"   ✅ Блок #{len(valid_detections)}: '{validated_text}' (валидировано)")
            else:
                print(f"   🗑️ Пропущен блок (пустой текст после валидации)")

    print(f"   Итоговое количество блоков после фильтрации: {len(valid_detections)}")

    return JSONResponse({
        "filename": filename,
        "url": f"/uploads/{filename}",
        "detections": valid_detections
    })


@app.post("/save")
async def save_transcriptions(data: Dict[str, Any]):
    """Сохранение транскрипций в текстовый файл"""
    transcriptions = data.get("transcriptions", [])
    image_name = data.get("image_name", "unknown")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"transcription_{timestamp}.txt"
    output_path = os.path.join("uploads", output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("РАСПОЗНАВАНИЕ ТЕКСТА\n")
        f.write("=" * 60 + "\n")
        f.write(f"Изображение: {image_name}\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("ДОПУСТИМЫЕ СИМВОЛЫ: а-я, А-Я, 0-9, N, V, X, !, ?, ., ,, :, ;, ), (, `, \", \', пробелы, дефис\n")
        f.write("-" * 60 + "\n\n")

        transcriptions.sort(key=lambda x: x.get('order', 0))

        f.write("РАСПОЗНАННЫЙ ТЕКСТ (в порядке чтения):\n")
        f.write("-" * 60 + "\n\n")

        for i, item in enumerate(transcriptions, 1):
            marker = "📎 " if item.get('is_merged') else "📄 "
            validated_marker = "✓ " if item.get('was_validated') else ""
            f.write(f"{marker}{validated_marker}[{i}] {item['text']}\n")
            if item.get('edited', False):
                f.write(f"    (отредактировано: было '{item['original']}')\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Всего строк: {len(transcriptions)}\n")
        f.write("=" * 60 + "\n")

    return JSONResponse({
        "success": True,
        "file": output_filename,
        "path": output_path
    })


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачивание файла с транскрипцией"""
    file_path = os.path.join("uploads", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="Файл не найден")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)