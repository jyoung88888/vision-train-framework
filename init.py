"""
init.py - 새 프로젝트 초기화 스크립트
========================================
역할:
  1. 프로젝트 폴더 구조 자동 생성
  2. DVC 초기화 + 리모트 서버 연결
  3. class_names가 적용된 config 파일 자동 생성
  4. .gitignore 설정

사용 예:
  python init.py --project 새프로젝트 --classes person helmet hoist
  python init.py --project 새프로젝트 --classes person --dvc-remote ssh://user@192.168.0.31:/mnt/hdd/data/새프로젝트
  python init.py --help
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# 초기화할 때 생성할 폴더 목록
_DIRS = [
    "data/images",
    "data/labels",
    "data/splits",
    "models",
    "mlflow_runs",
]

# config 템플릿 (class_names, project name이 치환됨)
_CONFIG_TEMPLATE = {
    "yolo.yaml": {
        "experiment": {
            "name": "{project}_yolo11m",
            "description": "{project} YOLOv11m 기본 학습",
            "tags": ["{project}", "yolo11m"],
        },
        "framework": "ultralytics",
        "model": {
            "architecture": "yolo11m.pt",
            "num_classes": "{num_classes}",
            "class_names": "{class_names}",
        },
        "data": {
            "task": "detect",
            "dataset_yaml": "data/dataset.yaml",
            "image_size": 640,
            "split_ratio": {"train": 0.8, "val": 0.15, "test": 0.05},
        },
        "train": {
            "epochs": 100,
            "batch_size": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "patience": 20,
            "device": "0",
            "workers": 8,
            "seed": 42,
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0,
            "perspective": 0.0,
            "flipud": 0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "mixup": 0,
            "copy_paste": 0,
            "erasing": 0.4,
        },
        "mlflow": {
            "tracking_uri": "./mlflow_runs",
            "experiment_name": "{project}-object-detection",
        },
        "dvc": {
            "remote": "thub",
        },
        "inference": {
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_det": 100,
            "half": True,
        },
    },
}


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """쉘 명령을 실행하고 결과를 반환한다."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}")
        sys.exit(1)
    return result


def create_dirs(project_root: Path) -> None:
    """프로젝트 폴더 구조를 생성한다."""
    print("\n[1/4] 폴더 구조 생성")
    for d in _DIRS:
        path = project_root / d
        path.mkdir(parents=True, exist_ok=True)
        # git이 빈 폴더를 추적하도록 .gitkeep 생성
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"  ✓ {d}/")


def init_dvc(project_root: Path, dvc_remote: str | None, project_name: str) -> None:
    """DVC를 초기화하고 리모트 서버를 연결한다."""
    print("\n[2/4] DVC 초기화")

    dvc_dir = project_root / ".dvc"
    if dvc_dir.exists():
        print("  이미 DVC가 초기화되어 있습니다.")
    else:
        run("dvc init")
        print("  ✓ DVC 초기화 완료")

    if dvc_remote:
        run(f'dvc remote add -d thub "{dvc_remote}"')
        print(f"  ✓ DVC 리모트 연결: {dvc_remote}")
    else:
        remote_path = f"ssh://user@192.168.0.31:/mnt/hdd/data/{project_name}"
        print(f"  [SKIP] --dvc-remote 미지정. 나중에 직접 설정하세요:")
        print(f"         dvc remote add -d thub \"{remote_path}\"")


def create_configs(project_root: Path, project_name: str, class_names: list[str]) -> None:
    """class_names와 project_name이 적용된 config 파일을 생성한다."""
    print("\n[3/4] Config 파일 생성")
    configs_dir = project_root / "configs"
    configs_dir.mkdir(exist_ok=True)

    num_classes = len(class_names)

    for filename, template in _CONFIG_TEMPLATE.items():
        config_path = configs_dir / filename

        # 템플릿 문자열 치환
        config_str = yaml.dump(template, default_flow_style=False, allow_unicode=True)
        config_str = (
            config_str
            .replace("'{project}'", project_name)
            .replace("{project}", project_name)
            .replace("'{num_classes}'", str(num_classes))
            .replace("{num_classes}", str(num_classes))
            .replace("'{class_names}'", str(class_names))
        )
        config = yaml.safe_load(config_str)

        # class_names 직접 치환
        config["model"]["class_names"] = class_names
        config["model"]["num_classes"] = num_classes

        # experiment 이름 치환
        config["experiment"]["name"] = f"{project_name}_yolo11m"
        config["experiment"]["description"] = f"{project_name} YOLOv11m 기본 학습"
        config["experiment"]["tags"] = [project_name, "yolo11m"]
        config["mlflow"]["experiment_name"] = f"{project_name}-object-detection"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"  ✓ configs/{filename}")


def print_next_steps(project_name: str, class_names: list[str]) -> None:
    """초기화 완료 후 다음 단계를 안내한다."""
    print("\n" + "=" * 55)
    print(f"  프로젝트 '{project_name}' 초기화 완료!")
    print("=" * 55)
    print(f"\n  클래스: {class_names}")
    print("\n  다음 단계:")
    print("  1. data/images/  에 이미지 추가")
    print("  2. data/labels/  에 라벨 추가 (YOLO txt 포맷)")
    print("  3. dvc add data/images data/labels")
    print("  4. git add data/images.dvc data/labels.dvc .dvc/config")
    print("  5. git commit -m '데이터 v1: 초기 데이터셋'")
    print("  6. dvc push")
    print("  7. python src/dataset.py --config configs/yolo.yaml")
    print("  8. python src/train.py   --config configs/yolo.yaml")
    print()


def init(project_name: str, class_names: list[str], dvc_remote: str | None) -> None:
    project_root = Path(__file__).resolve().parent

    print(f"\n프로젝트 초기화: {project_name}")
    print(f"클래스: {class_names}")

    create_dirs(project_root)
    init_dvc(project_root, dvc_remote, project_name)
    create_configs(project_root, project_name, class_names)
    print("\n[4/4] 완료")
    print_next_steps(project_name, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="새 프로젝트 초기화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python init.py --project crane_safety --classes person helmet hoist
  python init.py --project fire_detection --classes fire smoke \\
                 --dvc-remote ssh://user@192.168.0.31:/mnt/hdd/data/fire_detection
        """,
    )
    parser.add_argument(
        "--project", type=str, required=True,
        help="프로젝트 이름 (영문, 언더스코어 사용 권장)"
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", required=True,
        help="클래스 이름 목록 (예: person helmet hoist)"
    )
    parser.add_argument(
        "--dvc-remote", type=str, default=None,
        help="DVC 리모트 서버 주소 (예: ssh://user@192.168.0.31:/mnt/hdd/data/프로젝트)"
    )
    args = parser.parse_args()
    init(args.project, args.classes, args.dvc_remote)
