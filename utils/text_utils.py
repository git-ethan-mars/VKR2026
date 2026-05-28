import re
from typing import Optional, Dict, List

# Допустимые символы: русские буквы, цифры, знаки препинания и кавычки
ALLOWED_CHARS_PATTERN = re.compile(r'[^а-яА-Я0-9NVX,!?.:;()`\'"\s-]')


def validate_and_filter_text(text: str) -> Optional[str]:
    """
    Валидация и фильтрация текста.
    Удаляет символы, не входящие в разрешенный алфавит.
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


def validate_batch_texts(detections: List[Dict]) -> List[Dict]:
    """Валидация текстов для всех детекций."""
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


def filter_empty_detections(detections: List[Dict]) -> List[Dict]:
    """Фильтрует детекции, у которых после валидации текст стал пустым."""
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