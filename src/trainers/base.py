"""
trainers/base.py - 학습 프레임워크 추상 베이스 클래스
====================================================
역할:
  1. 모든 학습 프레임워크(Ultralytics, Detectron2 등)의 공통 인터페이스 정의
  2. MLflow 연동 로직 (파라미터/메트릭/아티팩트 기록) 공유
  3. TrainResult 데이터클래스로 학습 결과 표준화
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import mlflow


# ──────────────────────────────────────────────
# 학습 결과 표준 구조체
# ──────────────────────────────────────────────
@dataclass
class TrainResult:
    """프레임워크에 무관한 학습 결과."""
    save_dir: Path | None = None
    metrics_dict: dict[str, float] = field(default_factory=dict)
    best_weight_path: Path | None = None


# ──────────────────────────────────────────────
# 추상 베이스 클래스
# ──────────────────────────────────────────────
class BaseTrainer(ABC):
    """모든 학습 프레임워크가 구현해야 하는 인터페이스.

    공통 MLflow 로직은 이 클래스에서 제공하고,
    프레임워크별 학습/평가/모델 로딩은 서브클래스가 구현한다.
    """

    def __init__(self, cfg: dict, project_root: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.task = cfg["data"].get("task", "detect")

    # ──────────────────────────────────────────
    # 추상 메서드 (서브클래스 필수 구현)
    # ──────────────────────────────────────────
    @abstractmethod
    def create_model(self):
        """모델 객체를 생성하여 반환한다."""

    @abstractmethod
    def train(self, model, run_name: str) -> TrainResult:
        """학습을 실행하고 TrainResult를 반환한다."""

    @abstractmethod
    def evaluate(self, model_path: Path, split: str = "test") -> dict[str, float]:
        """지정 split에 대해 평가를 수행하고 메트릭 dict를 반환한다."""

    @abstractmethod
    def get_metric_keys(self) -> list[str]:
        """이 프레임워크가 생성하는 메트릭 키 목록을 반환한다."""

    @abstractmethod
    def find_best_weight(self, model_dir: Path) -> Path:
        """모델 디렉토리에서 best weight 파일 경로를 반환한다."""

    # ──────────────────────────────────────────
    # MLflow 공통 메서드
    # ──────────────────────────────────────────
    def setup_mlflow(self) -> None:
        """MLflow tracking URI와 experiment를 설정한다."""
        tracking_uri = Path(self.cfg["mlflow"]["tracking_uri"]).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])

    def log_params(self) -> None:
        """config의 주요 파라미터를 MLflow에 기록한다."""
        cfg = self.cfg
        mlflow.log_param("task", self.task)
        mlflow.log_param("framework", cfg.get("framework", "ultralytics"))
        mlflow.log_param("model_architecture", cfg["model"]["architecture"])
        mlflow.log_param("num_classes", cfg["model"]["num_classes"])
        mlflow.log_param("image_size", cfg["data"]["image_size"])

        for key, val in cfg["train"].items():
            mlflow.log_param(f"train_{key}", val)

        for key, val in cfg["augmentation"].items():
            mlflow.log_param(f"aug_{key}", val)

        mlflow.log_param("conf_threshold", cfg["inference"]["conf_threshold"])
        if "iou_threshold" in cfg["inference"]:
            mlflow.log_param("iou_threshold", cfg["inference"]["iou_threshold"])

    def log_metrics(self, metrics: dict) -> None:
        """학습 결과 메트릭을 MLflow에 기록한다."""
        metric_keys = self.get_metric_keys()
        for key in metric_keys:
            if key in metrics:
                safe_key = key.replace("/", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(safe_key, float(metrics[key]))

    def log_artifacts(self, save_dir: Path | None) -> None:
        """best weight를 MLflow 아티팩트로 기록한다."""
        if save_dir is None:
            return
        best_weight = self.find_best_weight(save_dir)
        if best_weight.exists():
            mlflow.log_artifact(str(best_weight), artifact_path="weights")
        if save_dir:
            mlflow.set_tag("model_save_dir", str(save_dir))

    def save_run_info(
        self,
        save_dir: Path | None,
        run_id: str,
        metrics_dict: dict,
        config_path: str,
        extra: dict | None = None,
    ) -> None:
        """모델 폴더에 run_info.json을 저장한다 (MLflow run_id 역추적용)."""
        if save_dir is None or not save_dir.exists():
            return
        run_info = {
            "mlflow_run_id": run_id,
            "experiment_name": save_dir.name,
            "description": self.cfg["experiment"]["description"],
            "config_file": config_path,
            "framework": self.cfg.get("framework", "ultralytics"),
            "architecture": self.cfg["model"]["architecture"],
            "task": self.task,
            "metrics": {k: float(v) for k, v in metrics_dict.items()},
        }
        if extra:
            run_info.update(extra)
        with open(save_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)
