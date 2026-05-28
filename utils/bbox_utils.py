from typing import List, Dict
import numpy as np


def is_bbox_inside(inner_bbox: List[int], outer_bbox: List[int]) -> bool:
    """Проверяет, содержится ли один bounding box полностью внутри другого."""
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
    """Удаляет bounding boxes, которые полностью содержатся внутри других."""
    if len(detections) <= 1:
        return detections

    detections_copy = detections.copy()
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

            if is_bbox_inside(inner_bbox, outer_bbox):
                to_remove.add(j)
                print(f"   🗑️ Удален вложенный блок: inner={inner_bbox} внутри outer={outer_bbox}")

    filtered_detections = [det for idx, det in enumerate(detections_copy) if idx not in to_remove]
    filtered_detections.sort(key=lambda d: d.get("original_index", 0))

    return filtered_detections


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


def merge_nearby_bboxes(detections: List[Dict], horizontal_threshold: float = 1.5,
                       vertical_threshold: float = 0.5) -> List[Dict]:
    """Объединяет близко расположенные bounding boxes в строки."""
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