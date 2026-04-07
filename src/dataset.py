"""
dataset.py - 데이터셋 준비 및 분할
====================================
역할:
  1. raw 이미지 + YOLO txt 라벨을 train/val/test로 분할 (detect / pose)
  2. 클래스별 폴더 이미지를 train/val/test로 분할 (classify)
  3. Ultralytics용 dataset.yaml 자동 생성 (detect / pose)
  4. Detectron2용 COCO JSON annotation 자동 생성
  5. 데이터 통계 출력 (클래스 분포, 이미지 수)

사용 예:
  python src/dataset.py --config configs/yolo.yaml
  python src/dataset.py --config configs/yolo_pose.yaml
  python src/dataset.py --config configs/yolo_classify.yaml
  python src/dataset.py --config configs/detectron2.yaml
  python src/dataset.py --config configs/yolo.yaml --force   # 강제 재분할
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from utils import load_config


_SPLIT_META_FILE = ".split_meta.json"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _compute_split_hash(
    pairs: list[tuple],
    ratios: dict[str, float],
    seed: int,
) -> str:
    """파일 목록 + split 설정의 해시를 계산한다."""
    filenames = sorted(p[0].name for p in pairs)
    key = json.dumps({"files": filenames, "ratios": ratios, "seed": seed}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _should_skip(output_root: Path, current_hash: str) -> bool:
    """이전 분할과 동일한 데이터/설정이면 True를 반환한다."""
    meta_path = output_root / _SPLIT_META_FILE
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("hash") == current_hash
    except (json.JSONDecodeError, KeyError):
        return False


def _save_split_meta(output_root: Path, current_hash: str, num_pairs: int) -> None:
    """분할 메타 정보를 저장한다."""
    meta_path = output_root / _SPLIT_META_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"hash": current_hash, "num_pairs": num_pairs}, f)


# ──────────────────────────────────────────────
# Detection / Pose 데이터 수집
# ──────────────────────────────────────────────
def collect_pairs(raw_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """이미지-라벨 쌍을 수집한다. 라벨이 없는 이미지는 건너뛴다."""
    pairs = []
    for img_path in sorted(raw_dir.iterdir()):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"[WARN] 라벨 없음, 건너뜀: {img_path.name}")
    return pairs


# ──────────────────────────────────────────────
# Classification 데이터 수집
# ──────────────────────────────────────────────
def collect_classify_images(data_dir: Path) -> list[tuple[Path, str]]:
    """클래스별 폴더에서 (이미지 경로, 클래스명) 쌍을 수집한다.

    Expected structure:
        data_dir/
        ├── person/
        │   ├── img001.jpg
        │   └── ...
        ├── helmet/
        │   └── ...
        └── hoist/
            └── ...
    """
    pairs = []
    class_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not class_dirs:
        print(f"[ERROR] {data_dir} 아래에 클래스 폴더가 없습니다.")
        return pairs

    for class_dir in class_dirs:
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            pairs.append((img_path, class_name))

    return pairs


# ──────────────────────────────────────────────
# 데이터 분할 (공통)
# ──────────────────────────────────────────────
def split_dataset(
    pairs: list[tuple],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, list[tuple]]:
    """train / val / test로 분할한다."""
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def _link_or_copy(src: Path, dst: Path) -> None:
    """심볼릭 링크를 시도하고, 실패하면 복사한다 (Windows 권한 문제 대응)."""
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


# ──────────────────────────────────────────────
# Detection / Pose 데이터 복사
# ──────────────────────────────────────────────
def copy_split(
    split_data: dict[str, list[tuple[Path, Path]]],
    output_root: Path,
) -> None:
    """분할된 데이터를 images/ labels/ 하위로 링크(또는 복사)한다."""
    # 이전 splits 정리
    if output_root.exists():
        shutil.rmtree(output_root)
        print(f"[INFO] 기존 splits 삭제 → {output_root}")

    use_symlink = platform.system() != "Windows"

    for split_name, pairs in split_data.items():
        img_dir = output_root / split_name / "images"
        lbl_dir = output_root / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in pairs:
            if use_symlink:
                _link_or_copy(img_path, img_dir / img_path.name)
                _link_or_copy(lbl_path, lbl_dir / lbl_path.name)
            else:
                shutil.copy2(img_path, img_dir / img_path.name)
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

    method = "symlink" if use_symlink else "copy"
    print(f"[INFO] 데이터 분할 완료 ({method}) → {output_root}")
    for k, v in split_data.items():
        print(f"  {k}: {len(v)}장")


# ──────────────────────────────────────────────
# Classification 데이터 복사
# ──────────────────────────────────────────────
def copy_split_classify(
    split_data: dict[str, list[tuple[Path, str]]],
    output_root: Path,
) -> None:
    """분할된 Classification 데이터를 클래스별 폴더로 복사한다.

    Output structure:
        output_root/
        ├── train/
        │   ├── person/
        │   ├── helmet/
        │   └── hoist/
        ├── val/
        │   └── ...
        └── test/
            └── ...
    """
    if output_root.exists():
        shutil.rmtree(output_root)
        print(f"[INFO] 기존 splits 삭제 → {output_root}")

    use_symlink = platform.system() != "Windows"

    for split_name, pairs in split_data.items():
        for img_path, class_name in pairs:
            class_dir = output_root / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / img_path.name

            if use_symlink:
                _link_or_copy(img_path, dst)
            else:
                shutil.copy2(img_path, dst)

    method = "symlink" if use_symlink else "copy"
    print(f"[INFO] Classification 데이터 분할 완료 ({method}) → {output_root}")
    for k, v in split_data.items():
        print(f"  {k}: {len(v)}장")


# ──────────────────────────────────────────────
# dataset.yaml 생성 (Detection / Pose)
# ──────────────────────────────────────────────
def generate_dataset_yaml(
    output_root: Path,
    class_names: list[str],
    yaml_path: Path,
    kpt_shape: list | None = None,
) -> None:
    """Ultralytics 포맷의 dataset.yaml을 생성한다.

    Args:
        kpt_shape: Pose 모델용 키포인트 shape (예: [17, 3]). None이면 detection용.
    """
    # Ultralytics는 상대경로를 datasets_dir 기준으로 해석하므로
    # 서버 간 호환을 위해 절대경로 사용
    abs_path = str(output_root.resolve())
    dataset_cfg = {
        "path": abs_path,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }

    # Pose 모델인 경우 kpt_shape 추가
    if kpt_shape is not None:
        dataset_cfg["kpt_shape"] = kpt_shape

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"[INFO] dataset.yaml 생성 → {yaml_path}")


# ──────────────────────────────────────────────
# 클래스 분포 출력
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# COCO JSON 생성 (Detectron2용)
# ──────────────────────────────────────────────
def generate_coco_json(
    split_data: dict[str, list[tuple[Path, Path]]],
    class_names: list[str],
    output_root: Path,
) -> None:
    """YOLO format 라벨을 COCO JSON annotation으로 변환한다.

    각 split(train/val/test)마다 annotations.json을 생성하며,
    Detectron2의 register_coco_instances()에서 사용할 수 있다.

    Args:
        split_data: split_dataset()의 반환값 {split: [(img, lbl), ...]}
        class_names: 클래스 이름 리스트
        output_root: data/splits/ 경로
    """
    for split_name, pairs in split_data.items():
        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(class_names)
            ],
        }

        ann_id = 0
        for img_id, (img_path, lbl_path) in enumerate(pairs):
            # 실제 이미지 크기 읽기 (YOLO normalized → COCO absolute 변환에 필요)
            img_file = output_root / split_name / "images" / img_path.name
            try:
                with Image.open(img_file) as img:
                    img_w, img_h = img.size
            except Exception:
                # 이미지를 읽을 수 없으면 skip
                print(f"[WARN] 이미지 읽기 실패, skip: {img_file}")
                continue

            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            })

            # YOLO label 파싱 및 COCO bbox 변환
            lbl_file = output_root / split_name / "labels" / lbl_path.name
            if not lbl_file.exists():
                continue

            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    # YOLO: cx_norm, cy_norm, w_norm, h_norm
                    cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
                    # COCO: x_abs, y_abs, w_abs, h_abs (top-left corner)
                    w_abs = w_n * img_w
                    h_abs = h_n * img_h
                    x_abs = cx_n * img_w - w_abs / 2
                    y_abs = cy_n * img_h - h_abs / 2

                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": [round(x_abs, 2), round(y_abs, 2),
                                 round(w_abs, 2), round(h_abs, 2)],
                        "area": round(w_abs * h_abs, 2),
                        "iscrowd": 0,
                    })
                    ann_id += 1

        # JSON 저장
        json_path = output_root / split_name / "annotations.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)

        print(f"[INFO] COCO JSON 생성: {json_path} "
              f"({len(coco['images'])} images, {len(coco['annotations'])} annotations)")


def print_class_distribution(
    split_data: dict[str, list[tuple[Path, Path]]],
    class_names: list[str],
) -> None:
    """각 split별 클래스 분포를 출력한다 (Detection / Pose)."""
    for split_name, pairs in split_data.items():
        counter = Counter()
        for _, lbl_path in pairs:
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
        print(f"\n[{split_name}] 클래스 분포:")
        for cls_id, name in enumerate(class_names):
            print(f"  {name}: {counter.get(cls_id, 0)}")


def print_class_distribution_classify(
    split_data: dict[str, list[tuple[Path, str]]],
) -> None:
    """각 split별 클래스 분포를 출력한다 (Classification)."""
    for split_name, pairs in split_data.items():
        counter = Counter(class_name for _, class_name in pairs)
        print(f"\n[{split_name}] 클래스 분포:")
        for class_name, count in sorted(counter.items()):
            print(f"  {class_name}: {count}")


# ──────────────────────────────────────────────
# 메인 prepare 함수
# ──────────────────────────────────────────────
def prepare(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    project_root = Path(__file__).resolve().parent.parent
    task = cfg["data"].get("task", "detect")

    output_root = project_root / "data" / "splits"

    if task == "classify":
        _prepare_classify(cfg, project_root, output_root, force)
    else:
        _prepare_detect(cfg, project_root, output_root, config_path, force)


def _prepare_detect(
    cfg: dict,
    project_root: Path,
    output_root: Path,
    config_path: str,
    force: bool,
) -> None:
    """Detection / Pose 데이터셋을 준비한다."""
    image_dir = project_root / "data" / "images"
    label_dir = project_root / "data" / "labels"
    yaml_path = project_root / cfg["data"]["dataset_yaml"]

    pairs = collect_pairs(image_dir, label_dir)
    if not pairs:
        print("[ERROR] 이미지-라벨 쌍이 없습니다. data/images 와 data/labels 를 확인하세요.")
        return

    print(f"[INFO] 총 {len(pairs)}개 이미지-라벨 쌍 발견")

    # 데이터/설정 변경 여부 확인 → 변경 없으면 skip
    split_hash = _compute_split_hash(pairs, cfg["data"]["split_ratio"], cfg["train"]["seed"])
    if not force and _should_skip(output_root, split_hash):
        print("[INFO] 데이터/설정 변경 없음 → 분할 skip (강제 재분할: --force)")
        kpt_shape = None
        if "keypoint" in cfg:
            kpt_shape = cfg["keypoint"]["kpt_shape"]
        generate_dataset_yaml(output_root, cfg["model"]["class_names"], yaml_path, kpt_shape)
        # Detectron2 COCO JSON이 없으면 생성
        if cfg.get("framework") == "detectron2":
            coco_missing = not (output_root / "train" / "annotations.json").exists()
            if coco_missing:
                print("[INFO] COCO JSON이 없어 생성합니다...")
                split_data = split_dataset(pairs, cfg["data"]["split_ratio"], cfg["train"]["seed"])
                generate_coco_json(split_data, cfg["model"]["class_names"], output_root)
        return

    split_data = split_dataset(pairs, cfg["data"]["split_ratio"], cfg["train"]["seed"])
    copy_split(split_data, output_root)
    _save_split_meta(output_root, split_hash, len(pairs))

    kpt_shape = None
    if "keypoint" in cfg:
        kpt_shape = cfg["keypoint"]["kpt_shape"]

    generate_dataset_yaml(output_root, cfg["model"]["class_names"], yaml_path, kpt_shape)

    # Detectron2 framework인 경우 COCO JSON도 함께 생성
    if cfg.get("framework") == "detectron2":
        generate_coco_json(split_data, cfg["model"]["class_names"], output_root)

    print_class_distribution(split_data, cfg["model"]["class_names"])


def _prepare_classify(
    cfg: dict,
    project_root: Path,
    output_root: Path,
    force: bool,
) -> None:
    """Classification 데이터셋을 준비한다."""
    data_dir = project_root / "data" / "images"

    if not data_dir.exists():
        print(f"[ERROR] Classification 데이터 폴더가 없습니다: {data_dir}")
        print(f"        data/images/ 아래에 클래스별 폴더(person/, helmet/ 등)를 만들고 이미지를 넣으세요.")
        return

    pairs = collect_classify_images(data_dir)
    if not pairs:
        print(f"[ERROR] {data_dir} 아래에 이미지가 없습니다.")
        return

    class_names = sorted(set(cls for _, cls in pairs))
    print(f"[INFO] 총 {len(pairs)}개 이미지 발견 ({len(class_names)}개 클래스)")

    # 데이터/설정 변경 여부 확인
    split_hash = _compute_split_hash(pairs, cfg["data"]["split_ratio"], cfg["train"]["seed"])
    if not force and _should_skip(output_root, split_hash):
        print("[INFO] 데이터/설정 변경 없음 → 분할 skip (강제 재분할: --force)")
        return

    split_data = split_dataset(pairs, cfg["data"]["split_ratio"], cfg["train"]["seed"])
    copy_split_classify(split_data, output_root)
    _save_split_meta(output_root, split_hash, len(pairs))

    print_class_distribution_classify(split_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 분할 및 dataset.yaml 생성")
    parser.add_argument(
        "--config", type=str, default="configs/yolo.yaml", help="실험 설정 파일"
    )
    parser.add_argument(
        "--force", action="store_true", help="데이터 변경 없어도 강제 재분할"
    )
    args = parser.parse_args()
    prepare(args.config, force=args.force)
