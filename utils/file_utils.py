import os
import uuid
from datetime import datetime
from typing import Dict, Any


def save_upload_file(file_data: bytes, filename: str, upload_dir: str = "uploads") -> str:
    """Сохраняет загруженный файл и возвращает путь к нему"""
    os.makedirs(upload_dir, exist_ok=True)

    file_extension = os.path.splitext(filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)

    with open(file_path, "wb") as f:
        f.write(file_data)

    return unique_filename, file_path


def save_transcriptions_to_file(transcriptions: list, image_name: str, output_dir: str = "uploads") -> str:
    """Сохраняет транскрипции в текстовый файл"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"transcription_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)

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

    return output_filename, output_path