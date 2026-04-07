"""
train.py - 학습 래퍼 (MLflow 연동)
====================================
역할:
  1. config YAML 로드
  2. framework에 따라 적절한 Trainer 생성 (Ultralytics / Detectron2)
  3. MLflow run 시작 → 파라미터 기록
  4. 학습 실행 → 메트릭/아티팩트 MLflow에 기록

사용 예:
  python src/train.py --config configs/yolo.yaml
  python src/train.py --config configs/rtdetr.yaml
  python src/train.py --config configs/yolo_classify.yaml
  python src/train.py --config configs/detectron2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import mlflow
from trainers import create_trainer
from utils import load_config


def get_next_run_name(models_dir: Path, base_name: str) -> str:
    """models 폴더를 확인해서 다음 넘버링된 실험 이름을 반환한다.

    Ultralytics와 동일한 넘버링 로직:
      exp_rtdetr_l → exp_rtdetr_l2 → exp_rtdetr_l3 → ...
    """
    if not (models_dir / base_name).exists():
        return base_name
    n = 2
    while (models_dir / f"{base_name}{n}").exists():
        n += 1
    return f"{base_name}{n}"


def update_mlflow_run_name(run_id: str, new_name: str) -> None:
    """MLflow API를 통해 run name을 업데이트한다."""
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "mlflow.runName", new_name)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    task = cfg["data"].get("task", "detect")

    # Trainer 생성 (framework에 따라 자동 분기)
    trainer = create_trainer(cfg, project_root)

    # 모델 생성
    model = trainer.create_model()

    # 학습 전에 다음 넘버링 이름을 미리 계산
    models_dir = project_root / "models"
    base_name = cfg["experiment"]["name"]
    run_name = get_next_run_name(models_dir, base_name)
    print(f"[INFO] 실험 이름: {run_name} (task: {task}, framework: {cfg.get('framework', 'ultralytics')})")

    # MLflow run 시작
    trainer.setup_mlflow()
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id

    save_dir = None
    try:
        # 파라미터 기록
        trainer.log_params()
        mlflow.set_tags({
            "description": cfg["experiment"]["description"],
            "config_file": config_path,
            "task": task,
            "framework": cfg.get("framework", "ultralytics"),
        })

        # 학습 실행
        result = trainer.train(model, run_name)
        save_dir = result.save_dir

        # Ultralytics가 MLflow run을 닫았으면 같은 run_id로 다시 열기
        if mlflow.active_run() is None:
            mlflow.start_run(run_id=run_id)

        # 메트릭 기록
        if result.metrics_dict:
            trainer.log_metrics(result.metrics_dict)

        # 아티팩트 기록
        trainer.log_artifacts(result.save_dir)

        # run_info.json 저장
        trainer.save_run_info(
            result.save_dir, run_id, result.metrics_dict, config_path
        )

        print(f"[INFO] 학습 완료. MLflow run: {run_id}")
        if save_dir:
            print(f"[INFO] 모델 저장: {save_dir}")

    finally:
        mlflow.end_run()

    if save_dir:
        update_mlflow_run_name(run_id, save_dir.name)
        print(f"[INFO] MLflow run name 업데이트: {save_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 (MLflow 연동)")
    parser.add_argument(
        "--config", type=str, default="configs/yolo.yaml", help="실험 설정 파일"
    )
    args = parser.parse_args()
    train(args.config)
