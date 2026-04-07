"""
trainers - 학습 프레임워크 추상화 패키지
========================================
팩토리 함수 create_trainer()로 config의 framework 필드에 따라
적절한 Trainer 인스턴스를 생성한다.

사용 예:
  from trainers import create_trainer
  trainer = create_trainer(cfg, project_root)
"""

from __future__ import annotations

from pathlib import Path

from trainers.base import BaseTrainer, TrainResult


def create_trainer(cfg: dict, project_root: Path) -> BaseTrainer:
    """config의 framework 필드를 기반으로 적절한 Trainer를 생성한다.

    Args:
        cfg: load_config()로 로드한 실험 설정 dict
        project_root: 프로젝트 루트 경로 (experiment/)

    Returns:
        BaseTrainer 서브클래스 인스턴스

    framework 필드가 없으면 기본값 'ultralytics'를 사용한다.
    기존 config 파일과 완전히 하위 호환된다.
    """
    framework = cfg.get("framework", "ultralytics")

    if framework == "ultralytics":
        from trainers.ultralytics_trainer import UltralyticsTrainer
        return UltralyticsTrainer(cfg, project_root)

    if framework == "detectron2":
        from trainers.detectron2_trainer import Detectron2Trainer
        return Detectron2Trainer(cfg, project_root)

    raise ValueError(
        f"알 수 없는 framework: '{framework}'. "
        f"지원: 'ultralytics', 'detectron2'"
    )


__all__ = ["BaseTrainer", "TrainResult", "create_trainer"]
