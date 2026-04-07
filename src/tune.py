"""
tune.py - Optuna 하이퍼파라미터 튜닝 (MLflow 연동)
====================================================
역할:
  1. config YAML의 tune.search_space 기반으로 Optuna 탐색 공간 정의
  2. 각 trial에서 파라미터 제안 → 학습 → 메트릭 반환
  3. 모든 trial을 MLflow에 개별 run으로 기록
  4. 최적 파라미터를 config YAML로 export
  5. Optuna 시각화를 HTML로 저장

사용 예:
  python src/tune.py --config configs/tune/tune_yolo.yaml
  python src/tune.py --config configs/tune/tune_yolo.yaml --resume
  python src/tune.py --config configs/tune/tune_classify.yaml
  python src/tune.py --config configs/tune/tune_detectron2.yaml
"""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import mlflow
import optuna
import torch
import yaml

from trainers import create_trainer
from utils import load_config

# 프로젝트 루트 설정 (src/tune.py → experiment/)
_project_root = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────
# 탐색 공간 → 파라미터 제안
# ──────────────────────────────────────────────
# search_space의 각 파라미터가 config에서 어느 섹션에 속하는지 매핑
_PARAM_SECTION = {
    # 공통
    "architecture": "model",
    "lr0": "train",
    "lrf": "train",
    "batch_size": "train",
    "optimizer": "train",
    "warmup_epochs": "train",
    "weight_decay": "train",
    "momentum": "train",
    # Ultralytics 증강
    "mosaic": "augmentation",
    "mixup": "augmentation",
    "scale": "augmentation",
    "copy_paste": "augmentation",
    "erasing": "augmentation",
    "fliplr": "augmentation",
    "flipud": "augmentation",
    "hsv_h": "augmentation",
    "hsv_s": "augmentation",
    "hsv_v": "augmentation",
    "degrees": "augmentation",
    "translate": "augmentation",
    "close_mosaic": "augmentation",
    # Detectron2 고유
    "anchor_sizes": "detectron2",
    "roi_batch_size": "detectron2",
    "eval_period": "detectron2",
}


def suggest_params(
    trial: optuna.Trial,
    search_space: dict,
) -> dict[str, any]:
    """search_space 정의에 따라 Optuna trial에서 파라미터를 제안한다."""
    params = {}
    for name, spec in search_space.items():
        param_type = spec["type"]

        if param_type == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"알 수 없는 파라미터 타입: {param_type} (파라미터: {name})")

    return params


def override_config(
    cfg: dict,
    params: dict,
    tune_cfg: dict,
) -> dict:
    """기본 config에 Optuna가 제안한 파라미터를 override한다."""
    cfg = copy.deepcopy(cfg)

    # tune 전용 설정 override
    cfg["train"]["epochs"] = tune_cfg["epochs_per_trial"]
    cfg["train"]["patience"] = tune_cfg["patience_per_trial"]

    # 탐색된 파라미터 override
    for name, value in params.items():
        section = _PARAM_SECTION.get(name)
        if section is None:
            print(f"[WARN] 알 수 없는 파라미터 섹션: {name}, 무시합니다.")
            continue
        # 섹션이 없으면 생성 (예: detectron2 섹션)
        if section not in cfg:
            cfg[section] = {}
        cfg[section][name] = value

    return cfg


# ──────────────────────────────────────────────
# Objective 함수
# ──────────────────────────────────────────────
def _cleanup_gpu(model=None) -> None:
    """GPU 메모리를 정리한다. trial 간 OOM 방지."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    tune_cfg: dict,
    config_path: str,
) -> float:
    """단일 trial을 실행하고 최적화 대상 메트릭을 반환한다."""
    project_root = _project_root
    task = base_cfg["data"].get("task", "detect")

    # 1. 파라미터 제안
    params = suggest_params(trial, tune_cfg["search_space"])

    # 2. config override
    cfg = override_config(base_cfg, params, tune_cfg)

    # 3. trial 이름 생성
    base_name = base_cfg["experiment"]["name"]
    trial_name = f"{base_name}_trial_{trial.number:02d}"

    # 4. trial 시작 전 GPU 메모리 정리
    _cleanup_gpu()

    # 5. Trainer 생성 및 모델 생성
    trainer = create_trainer(cfg, project_root)
    model = trainer.create_model()

    # 6. MLflow run 시작
    trainer.setup_mlflow()
    run = mlflow.start_run(run_name=trial_name)
    run_id = run.info.run_id

    metric_value = 0.0

    try:
        # 파라미터 기록
        trainer.log_params()
        mlflow.set_tags({
            "description": cfg["experiment"]["description"],
            "config_file": config_path,
            "task": task,
            "framework": cfg.get("framework", "ultralytics"),
            "optuna_trial": str(trial.number),
            "optuna_study": base_name,
        })
        # 제안된 파라미터도 별도 태그로 기록
        for name, value in params.items():
            mlflow.set_tag(f"optuna_param_{name}", str(value))

        # 7. 학습 실행
        result = trainer.train(model, trial_name)

        # Ultralytics가 MLflow run을 닫았으면 다시 열기
        if mlflow.active_run() is None:
            mlflow.start_run(run_id=run_id)

        # 8. 메트릭 기록
        if result.metrics_dict:
            trainer.log_metrics(result.metrics_dict)

            # 최적화 대상 메트릭 추출
            target_metric = tune_cfg["metric"]
            if target_metric in result.metrics_dict:
                metric_value = float(result.metrics_dict[target_metric])

        # best 모델 아티팩트 기록
        trainer.log_artifacts(result.save_dir)

        # run_info.json 저장
        trainer.save_run_info(
            result.save_dir, run_id, result.metrics_dict, config_path,
            extra={
                "optuna_trial": trial.number,
                "optuna_params": params,
            },
        )

        mlflow.log_metric("optuna_trial_number", trial.number)
        mlflow.log_metric("optuna_value", metric_value)

        print(f"\n[Trial {trial.number:02d}] {tune_cfg['metric']} = {metric_value:.4f}")
        print(f"  params: {params}")

    except Exception as e:
        print(f"\n[Trial {trial.number:02d}] 학습 실패: {e}")
        raise optuna.TrialPruned(f"학습 실패: {e}")

    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        # trial 종료 후 GPU 메모리 해제 (다음 trial을 위해)
        _cleanup_gpu(model)

    return metric_value


# ──────────────────────────────────────────────
# 시각화 저장
# ──────────────────────────────────────────────
def save_visualizations(study: optuna.Study, output_dir: Path) -> None:
    """Optuna 시각화를 HTML 파일로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )

        # 최적화 히스토리
        fig = plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        print(f"[INFO] 저장: {output_dir / 'optimization_history.html'}")

        # 파라미터 중요도
        if len(study.trials) >= 2:
            fig = plot_param_importances(study)
            fig.write_html(str(output_dir / "param_importances.html"))
            print(f"[INFO] 저장: {output_dir / 'param_importances.html'}")

        # 슬라이스 플롯
        fig = plot_slice(study)
        fig.write_html(str(output_dir / "slice_plot.html"))
        print(f"[INFO] 저장: {output_dir / 'slice_plot.html'}")

        # 등고선 플롯 (파라미터 2개 이상일 때)
        param_names = list(study.best_trial.params.keys())
        if len(param_names) >= 2:
            fig = plot_contour(study)
            fig.write_html(str(output_dir / "contour_plot.html"))
            print(f"[INFO] 저장: {output_dir / 'contour_plot.html'}")

    except ImportError:
        print("[WARN] plotly가 설치되지 않아 시각화를 건너뜁니다.")
        print("       pip install plotly 로 설치할 수 있습니다.")
    except Exception as e:
        print(f"[WARN] 시각화 생성 중 오류: {e}")


# ──────────────────────────────────────────────
# 최적 config export
# ──────────────────────────────────────────────
def export_best_config(
    study: optuna.Study,
    base_cfg: dict,
    tune_cfg: dict,
    output_path: Path,
) -> None:
    """최적 trial의 파라미터를 반영한 완전한 config YAML을 생성한다."""
    best_params = study.best_trial.params
    cfg = override_config(base_cfg, best_params, tune_cfg)

    # tune 섹션 제거 (본 학습용 config이므로)
    cfg.pop("tune", None)

    # 본 학습 설정으로 복원
    cfg["train"]["epochs"] = base_cfg["train"]["epochs"]
    cfg["train"]["patience"] = base_cfg["train"]["patience"]

    # 실험 이름/설명 업데이트
    cfg["experiment"]["name"] = f"best_{base_cfg['experiment']['name']}"
    cfg["experiment"]["description"] = (
        f"Optuna 최적 파라미터 (trial {study.best_trial.number}, "
        f"{tune_cfg['metric']}={study.best_value:.4f})"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAML 저장 시 best_params 주석 추가
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ============================================================\n")
        f.write("# Optuna 최적 파라미터로 생성된 config\n")
        f.write(f"# Best trial: {study.best_trial.number}\n")
        f.write(f"# Best {tune_cfg['metric']}: {study.best_value:.4f}\n")
        f.write("# ============================================================\n")
        f.write(f"# Best params:\n")
        for k, v in best_params.items():
            f.write(f"#   {k}: {v}\n")
        f.write("# ============================================================\n\n")
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[INFO] 최적 config 저장: {output_path}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def run_tuning(config_path: str, resume: bool = False) -> None:
    """Optuna study를 생성/재개하여 하이퍼파라미터 튜닝을 실행한다."""
    base_cfg = load_config(config_path)
    tune_cfg = base_cfg.get("tune")

    if tune_cfg is None:
        print("[ERROR] config에 'tune' 섹션이 없습니다.")
        print("        configs/tune/ 아래 튜닝 설정 파일을 사용하세요.")
        return

    project_root = _project_root
    study_name = base_cfg["experiment"]["name"]
    study_db = project_root / tune_cfg["study_db"]
    storage = f"sqlite:///{study_db}"

    # Study 생성 또는 재개
    if resume:
        print(f"[INFO] 기존 study 재개: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        completed = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = tune_cfg["n_trials"] - completed
        print(f"[INFO] 완료된 trial: {completed}, 남은 trial: {remaining}")
        if remaining <= 0:
            print("[INFO] 모든 trial이 완료되었습니다.")
            _print_results(study, base_cfg, tune_cfg, config_path)
            return
        n_trials = remaining
    else:
        print(f"[INFO] 새 study 생성: {study_name}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=tune_cfg["direction"],
            load_if_exists=False,
        )
        n_trials = tune_cfg["n_trials"]

    framework = base_cfg.get("framework", "ultralytics")
    print(f"[INFO] 총 {n_trials} trials 실행 예정 (framework: {framework})")
    print(f"[INFO] trial당 epochs: {tune_cfg['epochs_per_trial']}, "
          f"patience: {tune_cfg['patience_per_trial']}")
    print(f"[INFO] 최적화 대상: {tune_cfg['metric']} ({tune_cfg['direction']})")
    print(f"[INFO] Study DB: {study_db}")
    print("-" * 60)

    # 최적화 실행
    study.optimize(
        lambda trial: objective(trial, base_cfg, tune_cfg, config_path),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    _print_results(study, base_cfg, tune_cfg, config_path)


def _print_results(
    study: optuna.Study,
    base_cfg: dict,
    tune_cfg: dict,
    config_path: str,
) -> None:
    """튜닝 결과를 출력하고 시각화/config를 저장한다."""
    project_root = _project_root

    # 결과 출력
    print("\n" + "=" * 60)
    print("OPTUNA 튜닝 완료")
    print("=" * 60)

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n완료된 trials: {len(completed_trials)} / {len(study.trials)}")

    print(f"\n{'='*60}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best {tune_cfg['metric']}: {study.best_value:.4f}")
    print(f"{'='*60}")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # 상위 5개 trial 출력
    sorted_trials = sorted(
        completed_trials,
        key=lambda t: t.value if t.value is not None else float("-inf"),
        reverse=(tune_cfg["direction"] == "maximize"),
    )
    print(f"\n{'='*60}")
    print("Top 5 Trials:")
    print(f"{'Trial':>6} {'Value':>10}  Parameters")
    print("-" * 60)
    for t in sorted_trials[:5]:
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        value_str = f"{t.value:.4f}" if t.value is not None else "N/A"
        print(f"  #{t.number:<4} {value_str:>10}  {params_str}")

    # 시각화 저장
    viz_dir = project_root / "tune_results" / base_cfg["experiment"]["name"]
    save_visualizations(study, viz_dir)

    # 최적 config export
    best_config_path = (
        project_root / "configs" / f"best_params_{base_cfg['experiment']['name']}.yaml"
    )
    export_best_config(study, base_cfg, tune_cfg, best_config_path)

    print(f"\n{'='*60}")
    print("다음 단계:")
    print(f"  1. 시각화 확인: {viz_dir}/")
    print(f"  2. 최적 파라미터로 본 학습:")
    print(f"     python src/train.py --config {best_config_path.relative_to(project_root)}")
    print(f"  3. MLflow UI에서 전체 비교:")
    print(f"     mlflow ui --backend-store-uri ./mlflow_runs")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna 하이퍼파라미터 튜닝 (MLflow 연동)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="튜닝 설정 파일 (예: configs/tune/tune_yolo.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="이전 study를 이어서 실행",
    )
    args = parser.parse_args()
    run_tuning(args.config, resume=args.resume)
