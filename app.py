import os
from typing import Dict, Any

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from modules.image_preprocessor import ImagePreprocessor
from modules.trocr_recognizer import TrOCRRecognizer

from modules.yolo_detector import YOLODetector
from utils.bbox_utils import remove_nested_bboxes, merge_nearby_bboxes
from utils.file_utils import save_upload_file, save_transcriptions_to_file
from utils.text_utils import validate_and_filter_text

# Создаем необходимые директории
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = FastAPI(title="OCR Web App", description="Распознавание текста с помощью YOLO + TrOCR")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Инициализация компонентов
yolo_detector = YOLODetector()
trocr_recognizer = TrOCRRecognizer()
preprocessor = ImagePreprocessor()


@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при старте"""
    print("🚀 Запуск OCR приложения...")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    # Здесь HTML код из оригинального файла (можно вынести в отдельный файл)
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Загрузка изображения и распознавание"""
    # Сохраняем файл
    content = await file.read()
    filename, file_path = save_upload_file(content, file.filename)

    print(f"\n📊 Обработка изображения: {filename}")

    # ШАГ 1: Детекция bounding boxes с помощью YOLO
    raw_detections = yolo_detector.detect(file_path)
    print(f"   Найдено блоков до фильтрации: {len(raw_detections)}")

    # ШАГ 2: Предобработка bounding boxes (удаление вложенных)
    detections = remove_nested_bboxes(raw_detections)
    print(f"   После удаления вложенных блоков: {len(detections)}")

    # ШАГ 3: Объединение близких блоков в строки
    detections = merge_nearby_bboxes(detections)
    print(f"   После объединения в строки: {len(detections)}")

    # Загружаем изображение
    img = cv2.imread(file_path)
    img_h, img_w = img.shape[:2]

    valid_detections = []

    # ШАГ 4: Распознавание текста для каждого блока
    for idx, detection in enumerate(detections):
        # Вырезаем область
        crop = preprocessor.crop_bbox(img, detection["bbox"])

        if crop is not None:
            # Распознаем текст
            text = trocr_recognizer.recognize(crop, preprocessor.preprocess_for_trocr)

            # Валидируем текст
            validated_text = validate_and_filter_text(text)

            if validated_text:
                detection["text"] = validated_text
                detection["was_validated"] = True
                detection["order"] = len(valid_detections) + 1
                valid_detections.append(detection)

                if detection.get("is_merged"):
                    print(f"   ✅ Строка #{len(valid_detections)}: '{validated_text}' (объединено)")
                else:
                    print(f"   ✅ Блок #{len(valid_detections)}: '{validated_text}'")
            else:
                print(f"   🗑️ Пропущен блок (пустой текст после валидации)")

    print(f"   Итоговое количество блоков: {len(valid_detections)}")

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

    output_filename, output_path = save_transcriptions_to_file(transcriptions, image_name)

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