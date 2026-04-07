"""
eval.py - Test 데이터셋 평가 (MLflow 연동)
============================================
역할:
  1. 학습 완료된 모델(best weight)로 test 데이터셋 평가
  2. 평가 메트릭을 기존 MLflow run에 test_ prefix로 기록
  3. run_info.json에 test_metrics 추가

사용 예:
  python src/eval.py --config configs/rtdetr.yaml                  # 최근 실험 자동 선택
  python src/eval.py -e exp_rtdetr_l2 --config configs/rtdetr.yaml # 실험 지정
  python src/eval.py --config configs/yolo_classify.yaml           # Classification 평가
  python src/eval.py --config configs/detectron2.yaml              # Detectron2 평가
  python src/eval.py --list                                        # 실험 목록 출력
  
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from trainers import create_trainer
from utils import load_config


def setup_mlflow(cfg: dict) -> None:
    tracking_uri = Path(cfg["mlflow"]["tracking_uri"]).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def list_experiments(models_dir: Path) -> list[dict]:
    """models/ 하위의 실험 목록을 반환한다 (최신순 정렬)."""
    experiments = []
    if not models_dir.exists():
        return experiments
    for exp_dir in models_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        # Ultralytics (.pt) 또는 Detectron2 (.pth) weight 파일 탐색
        best_pt = exp_dir / "weights" / "best.pt"
        model_final = exp_dir / "model_final.pth"
        weight_file = best_pt if best_pt.exists() else model_final
        if not weight_file.exists():
            continue
        info = {"name": exp_dir.name, "path": exp_dir, "mtime": weight_file.stat().st_mtime}
        run_info_path = exp_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path, "r", encoding="utf-8") as f:
                run_info = json.load(f)
            info["architecture"] = run_info.get("architecture", "?")
            info["config_file"] = run_info.get("config_file", "?")
            info["framework"] = run_info.get("framework", "ultralytics")
            info["evaluated"] = "test_metrics" in run_info
        else:
            info["architecture"] = "?"
            info["config_file"] = "?"
            info["framework"] = "?"
            info["evaluated"] = False
        experiments.append(info)
    experiments.sort(key=lambda x: x["mtime"], reverse=True)
    return experiments


def print_experiment_list(models_dir: Path) -> None:
    """실험 목록을 출력한다."""
    experiments = list_experiments(models_dir)
    if not experiments:
        print("[INFO] models/ 에 평가 가능한 실험이 없습니다.")
        return
    print(f"\n{'#':<4} {'실험명':<30} {'모델':<20} {'프레임워크':<12} {'평가됨':<6} {'config'}")
    print("-" * 100)
    for i, exp in enumerate(experiments, 1):
        evaluated = "O" if exp["evaluated"] else "-"
        fw = exp.get("framework", "?")
        print(f"{i:<4} {exp['name']:<30} {exp['architecture']:<20} {fw:<12} {evaluated:<6} {exp['config_file']}")
    print(f"\n총 {len(experiments)}개 실험")


def find_latest_experiment(models_dir: Path) -> str | None:
    """가장 최근 실험 폴더명을 반환한다."""
    experiments = list_experiments(models_dir)
    return experiments[0]["name"] if experiments else None


def evaluate(experiment_name: str, config_path: str) -> None:
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    task = cfg["data"].get("task", "detect")
    framework = cfg.get("framework", "ultralytics")

    # Trainer 생성
    trainer = create_trainer(cfg, project_root)

    # 모델 디렉토리 확인
    model_dir = project_root / "models" / experiment_name
    best_weight = trainer.find_best_weight(model_dir)
    run_info_path = model_dir / "run_info.json"

    if not best_weight.exists():
        print(f"[ERROR] 모델 가중치 없음: {best_weight}")
        return

    # run_info.json에서 MLflow run_id 읽기
    run_id = None
    run_info = {}
    if run_info_path.exists():
        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        run_id = run_info.get("mlflow_run_id")
        print(f"[INFO] MLflow run_id: {run_id}")
    else:
        print("[WARN] run_info.json 없음 — MLflow 기록 없이 평가만 진행")

    print(f"[INFO] 모델 로드: {best_weight}")
    print(f"[INFO] test 데이터셋 평가 시작... (task: {task}, framework: {framework})")

    # test 데이터셋 평가 (Trainer가 프레임워크별 로직 처리)
    test_metrics = trainer.evaluate(best_weight, split="test")

    # 결과 출력
    print("\n[결과] Test 데이터셋 평가:")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}")

    # MLflow에 test 메트릭 기록 (기존 train run에 추가)
    if run_id:
        setup_mlflow(cfg)
        with mlflow.start_run(run_id=run_id):
            for key, val in test_metrics.items():
                safe_key = "test_" + key.replace("/", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(safe_key, val)
            mlflow.set_tag("test_evaluated", "true")
        print(f"[INFO] MLflow에 test 메트릭 기록 완료 (run: {run_id})")

    # run_info.json에 test_metrics 추가
    if run_info_path.exists():
        run_info["test_metrics"] = test_metrics
        with open(run_info_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)
        print(f"[INFO] run_info.json 업데이트 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 데이터셋 평가 (MLflow 연동)")
    parser.add_argument(
        "-e", "--experiment", type=str, default=None,
        help="실험 폴더명 (생략 시 가장 최근 실험 자동 선택)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="실험 설정 파일 (예: configs/rtdetr.yaml)"
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_exps",
        help="평가 가능한 실험 목록 출력"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    # --list: 목록만 출력하고 종료
    if args.list_exps:
        print_experiment_list(models_dir)
        raise SystemExit(0)

    # 실험명 자동 탐색
    experiment_name = args.experiment
    if experiment_name is None:
        experiment_name = find_latest_experiment(models_dir)
        if experiment_name is None:
            print("[ERROR] models/ 에 평가 가능한 실험이 없습니다.")
            raise SystemExit(1)
        print(f"[INFO] 최근 실험 자동 선택: {experiment_name}")

    # config 자동 탐색 (run_info.json에서)
    config_path = args.config
    if config_path is None:
        run_info_path = models_dir / experiment_name / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path, "r", encoding="utf-8") as f:
                config_path = json.load(f).get("config_file")
            if config_path:
                print(f"[INFO] config 자동 탐색: {config_path}")
        if config_path is None:
            print("[ERROR] --config를 지정하거나 run_info.json이 필요합니다.")
            raise SystemExit(1)

    evaluate(experiment_name, config_path)
