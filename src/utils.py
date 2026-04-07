"""
utils.py - 전처리 / 후처리 / 공통 유틸
=========================================
역할:
  1. 추론 결과 후처리 (NMS, 필터링, 구조화)
  2. 바운딩박스 유틸 (IoU, 좌표 변환)
  3. 시각화 헬퍼

** 서비스 연동 포인트 **
  운영 코드(rtsp_ai_trmp_Image_collection.py)에서도 이 모듈을
  import 하여 동일한 후처리 로직을 사용할 수 있도록 설계.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml


# ──────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# task별 메트릭 키 매핑
# ──────────────────────────────────────────────
METRIC_KEYS = {
    # Ultralytics
    "detect": [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ],
    "pose": [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ],
    "classify": [
        "metrics/accuracy_top1",
        "metrics/accuracy_top5",
    ],
    # Detectron2
    "detectron2_detect": [
        "bbox/AP",
        "bbox/AP50",
        "bbox/AP75",
        "bbox/APs",
        "bbox/APm",
        "bbox/APl",
    ],
    "detectron2_segment": [
        "bbox/AP",
        "bbox/AP50",
        "bbox/AP75",
        "segm/AP",
        "segm/AP50",
        "segm/AP75",
    ],
}


# ──────────────────────────────────────────────
# Detection 결과 구조체
# ──────────────────────────────────────────────
@dataclass
class Detection:
    """단일 검출 결과."""
    bbox: list[float]       # [x1, y1, x2, y2] (pixel)
    confidence: float
    class_id: int
    class_name: str = ""
    track_id: int | None = None


@dataclass
class FrameResult:
    """한 프레임의 전체 검출 결과."""
    detections: list[Detection] = field(default_factory=list)
    frame_id: int = 0
    timestamp: float = 0.0


# ──────────────────────────────────────────────
# 바운딩박스 유틸
# ──────────────────────────────────────────────
def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → [cx,cy,w,h]"""
    x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """[cx,cy,w,h] → [x1,y1,x2,y2]"""
    cx, cy, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """두 박스(xyxy)의 IoU를 계산한다."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────
# Ultralytics 결과 → Detection 변환
# ──────────────────────────────────────────────
def parse_yolo_results(
    results,
    class_names: list[str],
    conf_threshold: float = 0.5,
) -> FrameResult:
    """
    Ultralytics Results 객체를 FrameResult로 변환한다.

    Args:
        results: model.predict() 반환값의 단일 원소
        class_names: ["hoist", "helmet", "person"]
        conf_threshold: 최소 confidence
    """
    detections = []

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            if conf < conf_threshold:
                continue
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            detections.append(
                Detection(
                    bbox=bbox.tolist(),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=name,
                )
            )

    return FrameResult(detections=detections)


# ──────────────────────────────────────────────
# 클래스별 필터링
# ──────────────────────────────────────────────
def filter_by_class(
    frame_result: FrameResult,
    target_classes: list[str],
) -> list[Detection]:
    """특정 클래스만 필터링한다."""
    return [d for d in frame_result.detections if d.class_name in target_classes]


# ──────────────────────────────────────────────
# 근접도 계산 (서비스 공용)
# ──────────────────────────────────────────────
def compute_proximity(
    person_det: Detection,
    hoist_det: Detection,
    proximity_ratio: float = 0.3,
) -> bool:
    """사람 bbox가 호이스트 bbox에 일정 비율 이상 겹치는지 판별한다."""
    return compute_iou(
        np.array(person_det.bbox), np.array(hoist_det.bbox)
    ) >= proximity_ratio


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
_COLORS = {
    "hoist": (0, 165, 255),    # orange
    "helmet": (0, 255, 0),     # green
    "person": (255, 0, 0),     # blue
}


def draw_detections(
    frame: np.ndarray,
    frame_result: FrameResult,
    line_thickness: int = 2,
) -> np.ndarray:
    """프레임에 검출 결과를 그린다."""
    canvas = frame.copy()

    for det in frame_result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = _COLORS.get(det.class_name, (200, 200, 200))
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)
        cv2.putText(
            canvas, label, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    return canvas
