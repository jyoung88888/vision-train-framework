"""
trainers/detectron2_trainer.py - Detectron2 학습기
===================================================
역할:
  1. Detectron2 모델 생성 (model_zoo) 및 CfgNode 구성
  2. DefaultTrainer 기반 학습 실행
  3. COCOEvaluator 기반 평가
  4. COCO format 데이터셋 등록

지원 모델 예시:
  - Faster R-CNN:  COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
  - RetinaNet:     COCO-Detection/retinanet_R_50_FPN_3x.yaml
  - Mask R-CNN:    COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

from trainers.base import BaseTrainer, TrainResult
from utils import METRIC_KEYS


class Detectron2Trainer(BaseTrainer):
    """Detectron2 프레임워크 학습기."""

    def __init__(self, cfg: dict, project_root: Path) -> None:
        super().__init__(cfg, project_root)
        self._d2_cfg = None  # Detectron2 CfgNode (lazy init)

    # ──────────────────────────────────────────
    # Detectron2 CfgNode 빌드
    # ──────────────────────────────────────────
    def _build_d2_config(self, output_dir: str | None = None):
        """우리 YAML config를 Detectron2 CfgNode로 변환한다."""
        from detectron2 import model_zoo
        from detectron2.config import get_cfg

        cfg = self.cfg
        d2_section = cfg.get("detectron2", {})
        config_file = d2_section.get(
            "config_file", cfg["model"]["architecture"]
        )

        d2_cfg = get_cfg()
        d2_cfg.merge_from_file(model_zoo.get_config_file(config_file))

        # pretrained weights
        if d2_section.get("pretrained", True):
            d2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        # 데이터셋 등록 이름
        d2_cfg.DATASETS.TRAIN = ("dongyang_train",)
        d2_cfg.DATASETS.TEST = ("dongyang_val",)

        # 클래스 수
        num_classes = cfg["model"]["num_classes"]
        d2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        # RetinaNet 계열인 경우
        if "retinanet" in config_file.lower():
            d2_cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

        # Solver 매핑
        d2_cfg.SOLVER.BASE_LR = cfg["train"]["lr0"]
        d2_cfg.SOLVER.IMS_PER_BATCH = cfg["train"]["batch_size"]
        d2_cfg.SOLVER.MOMENTUM = cfg["train"]["momentum"]
        d2_cfg.SOLVER.WEIGHT_DECAY = cfg["train"]["weight_decay"]

        # epoch → iteration 변환
        # 실제 데이터셋 크기를 모르는 시점이므로 추정값 사용
        # 데이터셋 등록 후 정확한 값으로 재설정 가능
        estimated_dataset_size = d2_section.get("estimated_dataset_size", 1000)
        iters_per_epoch = max(
            1, estimated_dataset_size // cfg["train"]["batch_size"]
        )
        max_iter = cfg["train"]["epochs"] * iters_per_epoch
        d2_cfg.SOLVER.MAX_ITER = max_iter

        # warmup
        warmup_iters = cfg["train"]["warmup_epochs"] * iters_per_epoch
        d2_cfg.SOLVER.WARMUP_ITERS = warmup_iters
        d2_cfg.SOLVER.WARMUP_METHOD = "linear"

        # LR scheduler (cosine decay to lrf)
        d2_cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

        # 옵티마이저 매핑
        optimizer = cfg["train"].get("optimizer", "SGD").upper()
        if optimizer in ("ADAMW", "ADAM"):
            d2_cfg.SOLVER.OPTIMIZER = "ADAMW"
        else:
            d2_cfg.SOLVER.OPTIMIZER = "SGD"

        # 이미지 크기
        imgsz = cfg["data"]["image_size"]
        d2_cfg.INPUT.MIN_SIZE_TRAIN = (imgsz,)
        d2_cfg.INPUT.MAX_SIZE_TRAIN = imgsz
        d2_cfg.INPUT.MIN_SIZE_TEST = imgsz
        d2_cfg.INPUT.MAX_SIZE_TEST = imgsz

        # 디바이스
        device = str(cfg["train"]["device"])
        if device == "cpu":
            d2_cfg.MODEL.DEVICE = "cpu"
        else:
            d2_cfg.MODEL.DEVICE = "cuda"

        # DataLoader workers
        d2_cfg.DATALOADER.NUM_WORKERS = cfg["train"]["workers"]

        # 추론 threshold
        d2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg["inference"]["conf_threshold"]
        if "retinanet" in config_file.lower():
            d2_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = cfg["inference"]["conf_threshold"]

        # NMS IoU threshold
        if "iou_threshold" in cfg["inference"]:
            d2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = cfg["inference"]["iou_threshold"]

        # Detectron2 고유 설정
        if "anchor_sizes" in d2_section:
            d2_cfg.MODEL.ANCHOR_GENERATOR.SIZES = d2_section["anchor_sizes"]
        if "roi_batch_size" in d2_section:
            d2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = d2_section["roi_batch_size"]

        # 평가 주기
        eval_period = d2_section.get("eval_period", iters_per_epoch)
        d2_cfg.TEST.EVAL_PERIOD = eval_period

        # 출력 디렉토리
        if output_dir:
            d2_cfg.OUTPUT_DIR = output_dir

        # seed
        d2_cfg.SEED = cfg["train"]["seed"]

        d2_cfg.freeze()
        self._d2_cfg = d2_cfg
        return d2_cfg

    # ──────────────────────────────────────────
    # 데이터셋 등록
    # ──────────────────────────────────────────
    def _register_datasets(self) -> None:
        """COCO JSON 기반으로 Detectron2 데이터셋을 등록한다."""
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances

        splits_dir = self.project_root / "data" / "splits"

        for split_name, d2_name in [
            ("train", "dongyang_train"),
            ("val", "dongyang_val"),
            ("test", "dongyang_test"),
        ]:
            # 이미 등록되어 있으면 skip
            if d2_name in DatasetCatalog.list():
                continue

            json_path = str(splits_dir / split_name / "annotations.json")
            image_dir = str(splits_dir / split_name / "images")

            if not Path(json_path).exists():
                print(f"[WARN] COCO annotation 없음: {json_path}")
                print(f"       python src/dataset.py --config <config>.yaml 를 먼저 실행하세요.")
                continue

            register_coco_instances(d2_name, {}, json_path, image_dir)

            # 클래스 이름 설정
            class_names = self.cfg["model"].get("class_names", [])
            if class_names:
                MetadataCatalog.get(d2_name).set(thing_classes=class_names)

    def _get_dataset_size(self, dataset_name: str) -> int:
        """등록된 데이터셋의 크기를 반환한다."""
        from detectron2.data import DatasetCatalog
        try:
            return len(DatasetCatalog.get(dataset_name))
        except KeyError:
            return self.cfg.get("detectron2", {}).get("estimated_dataset_size", 1000)

    def _update_iterations(self, d2_cfg) -> None:
        """실제 데이터셋 크기로 iteration 수를 재계산한다."""
        dataset_size = self._get_dataset_size("dongyang_train")
        batch_size = self.cfg["train"]["batch_size"]
        iters_per_epoch = max(1, dataset_size // batch_size)

        # CfgNode가 frozen 상태이므로 defrost 후 수정
        d2_cfg.defrost()
        d2_cfg.SOLVER.MAX_ITER = self.cfg["train"]["epochs"] * iters_per_epoch
        d2_cfg.SOLVER.WARMUP_ITERS = (
            self.cfg["train"]["warmup_epochs"] * iters_per_epoch
        )
        d2_cfg.TEST.EVAL_PERIOD = self.cfg.get("detectron2", {}).get(
            "eval_period", iters_per_epoch
        )
        d2_cfg.freeze()

    # ──────────────────────────────────────────
    # 커스텀 Trainer (평가 hook 포함)
    # ──────────────────────────────────────────
    @staticmethod
    def _create_trainer_class():
        """DefaultTrainer를 확장하여 evaluator를 포함하는 클래스를 반환한다."""
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator

        class _D2Trainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                if output_folder is None:
                    output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
                return COCOEvaluator(
                    dataset_name, output_dir=output_folder
                )

        return _D2Trainer

    # ──────────────────────────────────────────
    # 추상 메서드 구현
    # ──────────────────────────────────────────
    def create_model(self):
        """Detectron2 config를 빌드한다 (모델은 Trainer 내부에서 생성됨)."""
        self._register_datasets()
        d2_cfg = self._build_d2_config()
        self._update_iterations(d2_cfg)
        return d2_cfg

    def train(self, model, run_name: str) -> TrainResult:
        """Detectron2 DefaultTrainer로 학습을 실행한다.

        Args:
            model: _build_d2_config()이 반환한 CfgNode 객체
            run_name: 실험 이름
        """
        d2_cfg = model  # create_model()이 CfgNode를 반환
        models_dir = self.project_root / "models"
        output_dir = str(models_dir / run_name)

        # output_dir 업데이트
        d2_cfg.defrost()
        d2_cfg.OUTPUT_DIR = output_dir
        d2_cfg.freeze()

        os.makedirs(output_dir, exist_ok=True)

        # 학습 실행
        TrainerClass = self._create_trainer_class()
        trainer = TrainerClass(d2_cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # 결과 수집
        save_dir = Path(output_dir)
        best_weight = save_dir / "model_final.pth"

        # 평가 결과 수집
        metrics_dict = {}
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        try:
            evaluator = COCOEvaluator(
                "dongyang_val", output_dir=str(save_dir / "eval")
            )
            val_loader = build_detection_test_loader(d2_cfg, "dongyang_val")
            results = inference_on_dataset(trainer.model, val_loader, evaluator)

            # Detectron2 결과를 flat dict로 변환
            if "bbox" in results:
                for k, v in results["bbox"].items():
                    metrics_dict[f"bbox/{k}"] = float(v)
            if "segm" in results:
                for k, v in results["segm"].items():
                    metrics_dict[f"segm/{k}"] = float(v)
        except Exception as e:
            print(f"[WARN] 학습 후 평가 중 오류: {e}")

        return TrainResult(
            save_dir=save_dir,
            metrics_dict=metrics_dict,
            best_weight_path=best_weight if best_weight.exists() else None,
        )

    def evaluate(self, model_path: Path, split: str = "test") -> dict[str, float]:
        """Detectron2 COCOEvaluator로 평가를 수행한다."""
        from detectron2.engine import DefaultPredictor
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        self._register_datasets()

        # split에 따른 데이터셋 이름
        dataset_map = {
            "train": "dongyang_train",
            "val": "dongyang_val",
            "test": "dongyang_test",
        }
        dataset_name = dataset_map.get(split, f"dongyang_{split}")

        # config 빌드
        output_dir = str(model_path.parent)
        d2_cfg = self._build_d2_config(output_dir=output_dir)

        # 모델 로드
        d2_cfg.defrost()
        d2_cfg.MODEL.WEIGHTS = str(model_path)
        d2_cfg.freeze()

        model = build_model(d2_cfg)
        DetectionCheckpointer(model).load(str(model_path))
        model.eval()

        # 평가 실행
        evaluator = COCOEvaluator(
            dataset_name, output_dir=str(Path(output_dir) / f"eval_{split}")
        )
        data_loader = build_detection_test_loader(d2_cfg, dataset_name)
        results = inference_on_dataset(model, data_loader, evaluator)

        # flat dict로 변환
        metrics = {}
        if "bbox" in results:
            for k, v in results["bbox"].items():
                metrics[f"bbox/{k}"] = float(v)
        if "segm" in results:
            for k, v in results["segm"].items():
                metrics[f"segm/{k}"] = float(v)

        return metrics

    def get_metric_keys(self) -> list[str]:
        task = self.task
        if task == "segment":
            return METRIC_KEYS.get(
                "detectron2_segment", METRIC_KEYS["detectron2_detect"]
            )
        return METRIC_KEYS.get("detectron2_detect", [])

    def find_best_weight(self, model_dir: Path) -> Path:
        return model_dir / "model_final.pth"
