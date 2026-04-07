"""
trainers/ultralytics_trainer.py - Ultralytics (YOLO / RT-DETR) 학습기
=====================================================================
역할:
  1. YOLO / RT-DETR 모델 생성 및 학습
  2. Ultralytics val() 기반 평가
  3. Ultralytics 고유 설정 (증강 필터, dataset.yaml 경로 등) 처리
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import RTDETR, YOLO, settings as ultra_settings

from trainers.base import BaseTrainer, TrainResult
from utils import METRIC_KEYS

# Classification에서 사용하지 않는 증강 파라미터
_DETECT_ONLY_AUGMENTS = {"mosaic", "mixup", "copy_paste", "close_mosaic"}


class UltralyticsTrainer(BaseTrainer):
    """Ultralytics 프레임워크 (YOLO, RT-DETR) 학습기."""

    def __init__(self, cfg: dict, project_root: Path) -> None:
        super().__init__(cfg, project_root)
        # Ultralytics 전역 설정
        ultra_settings.update({
            "mlflow": False,
            "datasets_dir": str(project_root),
        })

    # ──────────────────────────────────────────
    # 추상 메서드 구현
    # ──────────────────────────────────────────
    def create_model(self):
        """YOLO 또는 RT-DETR 모델을 생성한다."""
        arch = self.cfg["model"]["architecture"]
        if "rtdetr" in arch.lower():
            return RTDETR(arch)
        return YOLO(arch)

    def train(self, model, run_name: str) -> TrainResult:
        """Ultralytics model.train()을 실행한다."""
        train_args = self._build_train_args(run_name)
        results = model.train(**train_args)

        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
        metrics_dict = {}
        if hasattr(results, "results_dict"):
            metrics_dict = {k: float(v) for k, v in results.results_dict.items()}

        best_weight = None
        if save_dir:
            candidate = save_dir / "weights" / "best.pt"
            if candidate.exists():
                best_weight = candidate

        return TrainResult(
            save_dir=save_dir,
            metrics_dict=metrics_dict,
            best_weight_path=best_weight,
        )

    def evaluate(self, model_path: Path, split: str = "test") -> dict[str, float]:
        """Ultralytics model.val()로 평가를 수행한다."""
        cfg = self.cfg
        arch = cfg["model"]["architecture"]

        if "rtdetr" in arch.lower():
            model = RTDETR(str(model_path))
        else:
            model = YOLO(str(model_path))

        if self.task == "classify":
            data_path = str((self.project_root / "data" / "splits").resolve())
        else:
            data_path = str(self.project_root / cfg["data"]["dataset_yaml"])

        val_args = {
            "data": data_path,
            "split": split,
            "imgsz": cfg["data"]["image_size"],
            "batch": cfg["train"]["batch_size"],
            "conf": cfg["inference"]["conf_threshold"],
            "device": cfg["train"]["device"],
            "workers": cfg["train"]["workers"],
        }
        if self.task != "classify" and "iou_threshold" in cfg["inference"]:
            val_args["iou"] = cfg["inference"]["iou_threshold"]

        results = model.val(**val_args)

        metrics = {}
        if hasattr(results, "results_dict"):
            for key in self.get_metric_keys():
                if key in results.results_dict:
                    metrics[key] = float(results.results_dict[key])
        return metrics

    def get_metric_keys(self) -> list[str]:
        return METRIC_KEYS.get(self.task, METRIC_KEYS["detect"])

    def find_best_weight(self, model_dir: Path) -> Path:
        return model_dir / "weights" / "best.pt"

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────
    def _build_train_args(self, run_name: str) -> dict:
        """config에서 Ultralytics model.train() 인자를 구성한다."""
        cfg = self.cfg
        models_dir = self.project_root / "models"

        if self.task == "classify":
            data_path = str((self.project_root / "data" / "splits").resolve())
        else:
            data_path = str(self.project_root / cfg["data"]["dataset_yaml"])

        train_args = {
            "data": data_path,
            "epochs": cfg["train"]["epochs"],
            "batch": cfg["train"]["batch_size"],
            "imgsz": cfg["data"]["image_size"],
            "optimizer": cfg["train"]["optimizer"],
            "lr0": cfg["train"]["lr0"],
            "lrf": cfg["train"]["lrf"],
            "momentum": cfg["train"]["momentum"],
            "weight_decay": cfg["train"]["weight_decay"],
            "warmup_epochs": cfg["train"]["warmup_epochs"],
            "patience": cfg["train"]["patience"],
            "device": cfg["train"]["device"],
            "workers": cfg["train"]["workers"],
            "seed": cfg["train"]["seed"],
            "project": str(models_dir),
            "name": run_name,
        }

        for key, val in cfg["augmentation"].items():
            if self.task == "classify" and key in _DETECT_ONLY_AUGMENTS:
                continue
            train_args[key] = val

        return train_args
