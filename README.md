## Object Detection 연구/학습 파이프라인

**PyTorch + Ultralytics (YOLOv11 / RT-DETR) + DVC + MLflow + Optuna** 기반으로
데이터셋 버전 관리, 실험 추적, 하이퍼파라미터 자동 튜닝, 서비스 연동 가능한 공통 모듈을 제공한다.

---

## 폴더 구조

```
experiment/
├── configs/                     # 하이퍼파라미터 설정 (실험별 분기)
│   ├── yolo.yaml                #   YOLOv11 기본 실험 설정
│   ├── rtdetr.yaml              #   RT-DETR 실험 설정
│   ├── yolo_pose.yaml           #   YOLO Pose 실험 설정
│   ├── tune_yolo.yaml           #   Optuna YOLO 튜닝 설정
│   └── tune_rtdetr.yaml         #   Optuna RT-DETR 튜닝 설정
├── data/
│   ├── images/                  # 원본 이미지 (DVC 추적)
│   ├── labels/                  # YOLO txt 어노테이션 (DVC 추적)
│   ├── splits/                  # dataset.py가 생성하는 train/val/test (자동)
│   ├── dataset.yaml             # 데이터셋 경로+클래스 정보 (자동 생성, git 제외)
│   └── dvc.yaml
├── models/                   # 학습 결과 weights (MLflow 아티팩트)
├── mlflow_runs/              # 로컬 MLflow 저장소
├── tune_results/             # Optuna 시각화 HTML (자동 생성)
├── src/
│   ├── __init__.py
│   ├── dataset.py            # 데이터 분할 + dataset.yaml 자동 생성
│   ├── train.py              # 학습 래퍼 (MLflow 연동, 모델 종류 무관)
│   ├── eval.py               # Test 데이터셋 평가 (MLflow 연동)
│   ├── utils.py              # 후처리, bbox 유틸, 시각화 (서비스 공용)
│   └── tune.py              # Optuna 하이퍼파라미터 튜닝 (MLflow 연동)
├── dvc.yaml                  # DVC 파이프라인 정의
├── requirements.txt
├── .gitignore
└── README.md
```

### 각 파일 역할

| 파일 | 역할 |
|---|---|
| `configs/` | **하이퍼파라미터 설정**. 모델, lr, batch_size, 증강, MLflow, 추론 파라미터 일괄 관리 |
| ┣ `yolo.yaml` | YOLOv11 기본 학습 설정 (CNN 기반, 실시간 추론에 적합) |
| ┣ `rtdetr.yaml` | RT-DETR 학습 설정 (Transformer 기반, 낮은 lr, 작은 batch, 고해상도) |
| ┣ `yolo_pose.yaml` | YOLO Pose 학습 설정 (키포인트 추정, COCO 17 관절) |
| ┣ `tune_yolo.yaml` | Optuna YOLO 튜닝 설정 (모델 크기 + HP 탐색 공간 정의) |
| ┗ `tune_rtdetr.yaml` | Optuna RT-DETR 튜닝 설정 (Transformer 맞춤 탐색 범위) |
| `data/dataset.yaml` | **데이터셋 경로/클래스 정보** (자동 생성). `dataset.py`가 서버의 절대경로로 자동 생성하므로 직접 수정 불필요 |
| `src/dataset.py` | `data/images/` + `data/labels/` → train/val/test 분할, `data/dataset.yaml` 자동 생성, 클래스 분포 출력 |
| `src/train.py` | config 로드 → MLflow run 시작 → 학습 → 메트릭(mAP, precision, recall) + best.pt 아티팩트 기록 |
| `src/eval.py` | 학습 완료된 모델로 test 데이터셋 평가, MLflow에 test 메트릭 기록 |
| `src/utils.py` | `load_config` 공용 함수, `Detection`/`FrameResult` 구조체, 결과 파싱, bbox 변환(xyxy↔xywh), IoU, 근접도 판별, 시각화 |
| `src/tune.py` | Optuna 하이퍼파라미터 튜닝. config의 탐색 공간에서 최적 파라미터를 자동 탐색하고 MLflow에 기록 |
| `dvc.yaml` | `prepare` → `train` 2단계 파이프라인. `dvc repro` 한 번으로 전체 실행 |

---

## 지원 모델

Ultralytics 프레임워크를 통해 다양한 모델을 **config 변경만으로** 사용할 수 있다.

| 모델 | architecture 값 | 특징 |
|------|-----------------|------|
| YOLOv11 nano | `yolo11n.pt` | 가장 가벼움, 실시간 추론용 |
| YOLOv11 small | `yolo11s.pt` | 속도-정확도 균형 |
| YOLOv11 medium | `yolo11m.pt` | 범용 |
| YOLOv11 large | `yolo11l.pt` | 고정확도 |
| YOLOv11 xlarge | `yolo11x.pt` | 최고 정확도 |
| RT-DETR-L | `rtdetr-l.pt` | Transformer 기반, NMS 불필요, 작은 객체 우수 |
| RT-DETR-X | `rtdetr-x.pt` | RT-DETR 고정확도 버전 |
| YOLOv8/v9/v10 | `yolov8n.pt` 등 | Ultralytics 하위 호환 |

---

## 환경 설정

### 1. Python 환경

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Git + DVC 초기화

```bash
git init
dvc init
```

### 3. (선택) DVC 리모트 설정

```bash
# 로컬 스토리지
dvc remote add -d local ./dvc_storage

# S3
# dvc remote add -d s3remote s3://bucket-name/path
```

---

## 새 서버에서 시작하기 (Quick Start)

### 1. 코드 클론

```bash
git clone http://git.i-gns.co.kr/iljoo_ai_team/ml-experiment-pipeline.git
cd ml-experiment-pipeline
```

### 2. Python 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. DVC 데이터셋 다운로드

데이터셋은 DVC로 관리되며, 원격 스토리지(`ssh://192.168.0.13` = Thub 4090)에 저장되어 있다.

```bash
# 데이터 다운로드 (raw 이미지 + labels)
dvc pull
```

> **참고**: DVC remote 서버에 SSH 접속 권한이 필요하다.
> 접속이 안 되면 SSH 키를 먼저 등록한다:
> ```bash
> ssh-copy-id <user>@XXX.XXX.XXX.XXX
> ```

DVC remote를 변경해야 하는 경우:
```bash
# 현재 remote 확인
dvc remote list

# 새 remote로 변경
dvc remote add -d new_remote ssh://<user>@<새서버IP>:<경로>
dvc pull
```

### 4. 서버 환경에 맞게 config 수정

학습 전에 `configs/yolo.yaml` (또는 사용할 config)에서 서버 환경에 맞게 아래 항목을 확인/수정한다.

```yaml
# --- 반드시 확인 ---
train:
  device: "0"              # 사용할 GPU 번호 (nvidia-smi로 확인, 멀티 GPU: "0,1")
  batch_size: 16           # GPU VRAM에 따라 조절
                           #   8GB  → 8
                           #   16GB → 16
                           #   24GB → 32
  workers: 8               # CPU 코어 수에 따라 조절

# --- 선택 수정 ---
data:
  image_size: 640          # RT-DETR 사용 시 1280 권장

train:
  epochs: 100              # 학습 에폭 수
  patience: 20             # early stopping (val loss 개선 없으면 중단)
```

> **주의**: RT-DETR 모델(`configs/rtdetr.yaml`)은 VRAM을 많이 사용하므로
> `batch_size: 4`, `image_size: 1280`이 기본 설정이다.

### 5. 학습 실행

```bash
# 1) 데이터 분할 (반드시 먼저 실행)
python src/dataset.py --config configs/yolo.yaml

# 2) 학습
python src/train.py --config configs/yolo.yaml
```

> **참고**: `data/dataset.yaml`은 `dataset.py`가 **서버의 절대경로로 자동 생성**하는 파일이다.
> git에 포함되지 않으므로(.gitignore), 새 서버에서는 반드시 `dataset.py`를 먼저 실행해야 한다.

---

## 사용 방법

### Step 1. 데이터 준비

원본 이미지와 YOLO 포맷 라벨을 배치

```
data/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── labels/
    ├── img_001.txt       # class_id cx cy w h (normalized)
    ├── img_002.txt
    └── ...
```

라벨 포맷 (YOLO txt)
```
0 0.512 0.345 0.120 0.230    # person
1 0.231 0.456 0.050 0.060    # helmet
2 0.678 0.789 0.080 0.200    # hoist
```

클래스 ID 매핑
| ID | 클래스 |
|----|--------|
| 0 | person |
| 1 | helmet |
| 2 | hoist |

### Step 2. DVC로 데이터 버전 관리

```bash
dvc add data/images data/labels
git add data/images.dvc data/labels.dvc data/.gitignore
git commit -m "데이터 v1: 초기 데이터셋 500장"
```

데이터가 추가/변경될 때마다:
```bash
# 이미지 추가 후
dvc add data/images data/labels
git add data/images.dvc data/labels.dvc
git commit -m "데이터 v2: 야간 이미지 200장 추가"
dvc push                      # 리모트에 업로드
```

#### 과거 데이터 버전으로 돌아가기

DVC는 git commit과 `.dvc` 파일을 연동하여 데이터 버전을 관리한다.
특정 시점의 데이터로 되돌리려면 git checkout 후 `dvc checkout`을 실행한다.

```bash
# 1) 데이터 버전 히스토리 확인
git log --oneline data/images.dvc
# 예시 출력:
#   a1b2c3d 데이터 v3: 재검수 + 500장 추가
#   e4f5g6h 데이터 v2: 야간 이미지 200장 추가
#   i7j8k9l 데이터 v1: 초기 데이터셋 500장

# 2) 원하는 버전의 .dvc 파일 복원
git checkout e4f5g6h -- data/images.dvc data/labels.dvc

# 3) 실제 데이터 파일을 해당 버전으로 교체
dvc checkout
```

> **동작 원리**: `.dvc` 파일 안에 데이터의 md5 해시가 기록되어 있다.
> `git checkout`으로 `.dvc` 파일을 과거 버전으로 되돌리면,
> `dvc checkout`이 해당 해시에 맞는 데이터를 DVC 캐시에서 복원한다.
> 캐시에 없으면 `dvc pull`로 리모트에서 다운로드한다.

```bash
# 캐시에 없는 경우 리모트에서 다운로드
dvc pull

# 최신 버전으로 다시 돌아가기
git checkout main -- data/images.dvc data/labels.dvc
dvc checkout
```

### Step 3. 데이터셋 분할

```bash
python src/dataset.py --config configs/yolo.yaml
```

실행 결과:
```
[INFO] 총 500개 이미지-라벨 쌍 발견
[INFO] 데이터 분할 완료 → data/splits
  train: 400장
  val: 75장
  test: 25장
[INFO] dataset.yaml 생성 → data/dataset.yaml

[train] 클래스 분포:
  hoist: 412
  helmet: 1893
  person: 2105
```

### Step 4. 학습

```bash
# YOLOv11 학습
python src/train.py --config configs/yolo.yaml

# RT-DETR 학습
python src/train.py --config configs/rtdetr.yaml
```

학습이 끝나면:
- `models/<experiment_name>/weights/best.pt` 파일이 로컬에 생성됨
- MLflow에 파라미터 + 메트릭 + 모델 아티팩트(`best.pt`)가 자동 기록됨
- 운영 기준 모델 소스는 MLflow를 우선으로 사용

### Step 5. Test 데이터셋 평가

학습 완료 후 test 데이터셋으로 최종 성능을 평가한다.

```bash
python src/eval.py -e <실험폴더명> --config <사용한 config>

# 예시
python src/eval.py -e baseline_yolo11n --config configs/yolo.yaml
python src/eval.py -e exp_rtdetr_l2 --config configs/rtdetr.yaml
```

평가 결과는:
- 터미널에 test precision, recall, mAP 출력
- MLflow에 `test_` prefix로 메트릭 기록 (학습 run에 추가)
- `models/<실험폴더명>/run_info.json`에 `test_metrics` 추가

### Step 6. MLflow UI로 실험 비교

```bash
mlflow ui --backend-store-uri ./mlflow_runs
# 브라우저에서 http://127.0.0.1:5000 접속
```

YOLOv11과 RT-DETR의 mAP, precision, recall을 나란히 비교 가능.

### (대안) DVC 파이프라인으로 한 번에 실행

Step 3 + Step 4를 한 번에:
```bash
# 기본 (yolo.yaml)
dvc repro

# 다른 config로 실행
dvc repro -S config=configs/rtdetr.yaml
dvc repro -S config=configs/yolo_pose.yaml
```

단계별 실행:
```bash
dvc repro prepare    # 데이터 분할만
dvc repro train      # 학습만
```

---

## Optuna 하이퍼파라미터 튜닝 (HPO)

Optuna를 사용하여 모델 아키텍처, 학습률, 증강 파라미터 등을 **자동으로 탐색**하고 최적 조합을 찾는다.
모든 trial은 MLflow에 기록되어 기존 수동 실험과 동일하게 비교 가능하다.

### 동작 원리

```
Optuna Study (n_trials=20)
│
├── Trial 0: yolo11s, lr=0.003, mosaic=0.7 → mAP50=0.782
├── Trial 1: yolo11m, lr=0.0008, mosaic=0.3 → mAP50=0.801  (이전 결과 참고하여 탐색)
├── Trial 2: yolo11m, lr=0.0005, mosaic=0.1 → mAP50=0.823
│   ...
└── Trial 19: 최종 결과 집계
    ├── Best params → configs/best_params_tune_yolo11.yaml (본 학습용)
    ├── 시각화 HTML → tune_results/ (파라미터 중요도, 최적화 히스토리 등)
    └── MLflow → 모든 trial을 개별 run으로 기록
```

> Trial이 쌓일수록 Optuna(TPE 알고리즘)가 **유망한 영역을 집중 탐색**한다 (단순 랜덤 서치가 아님).

### 튜닝 실행

```bash
# 1) 데이터 분할 (아직 안 했다면)
python src/dataset.py --config configs/tune_yolo.yaml

# 2) YOLO 튜닝 실행
python src/tune.py --config configs/tune_yolo.yaml

# 3) RT-DETR 튜닝 실행
python src/tune.py --config configs/tune_rtdetr.yaml
```

### 중단 / 재개

튜닝은 SQLite DB에 저장되므로 중단 후 이어서 실행할 수 있다.

```bash
# Ctrl+C로 중단 후 재개
python src/tune.py --config configs/tune_yolo.yaml --resume
```

### 튜닝 결과물

튜닝이 완료되면 다음 파일들이 자동 생성된다.

```
experiment/
├── optuna_study.db                            # Study DB (중단/재개용)
├── tune_results/tune_yolo11/
│   ├── optimization_history.html              # trial별 mAP 추이 그래프
│   ├── param_importances.html                 # 파라미터 중요도 (어떤 파라미터가 성능에 영향이 큰지)
│   ├── slice_plot.html                        # 파라미터별 성능 분포
│   └── contour_plot.html                      # 파라미터 조합별 성능 등고선
├── configs/
│   └── best_params_tune_yolo11.yaml           # 최적 파라미터 config (train.py에서 바로 사용 가능)
└── models/
    ├── tune_yolo11_trial_00/weights/best.pt   # 각 trial의 모델
    ├── tune_yolo11_trial_01/weights/best.pt
    └── ...
```

### 최적 파라미터로 본 학습

튜닝에서는 epochs를 줄여서 빠르게 탐색하므로, 최적 파라미터를 찾은 뒤 full epochs로 본 학습을 실행한다.

```bash
# 자동 생성된 best config로 본 학습 (epochs, patience는 원래 값으로 복원되어 있음)
python src/train.py --config configs/best_params_tune_yolo11.yaml
```

### MLflow에서 튜닝 결과 확인

```bash
mlflow ui --backend-store-uri ./mlflow_runs
# 브라우저에서 http://127.0.0.1:5000 접속
```

MLflow UI에서 확인 가능한 내용:
- 모든 trial이 **개별 run**으로 기록됨 (`optuna_trial` 태그로 필터링 가능)
- trial별 mAP50, precision, recall **테이블 비교**
- 파라미터별 성능 **차트 비교**
- 각 trial의 best.pt 모델 **아티팩트 다운로드**

### 탐색 공간 커스터마이징

`configs/tune_yolo.yaml`의 `tune.search_space`를 수정하여 탐색 범위를 조절할 수 있다.

```yaml
tune:
  n_trials: 20                     # trial 수 조절
  epochs_per_trial: 50             # trial당 학습 에폭
  patience_per_trial: 10           # trial당 early stopping

  search_space:
    # 파라미터 추가/제거/범위 변경 가능
    lr0:
      type: float                  # float, int, categorical 지원
      low: 0.0001
      high: 0.01
      log: true                    # 로그 스케일 탐색 (학습률에 적합)
    batch_size:
      type: categorical
      choices: [8, 16, 32]
    architecture:
      type: categorical
      choices: ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
```

### 소요 시간 참고

| GPU | 1 trial (50 epochs) | 20 trials |
|-----|---------------------|-----------|
| RTX 4090 | ~30분~1시간 | ~10~20시간 |

> **팁**: 시간이 부담되면 `n_trials: 5~10`, `epochs_per_trial: 30`으로 먼저 경향을 파악한 뒤,
> 유망한 범위로 좁혀서 정밀 탐색하는 2단계 전략이 효율적이다.

---

## 실험 관리

### 새 실험 만들기

기존 config를 복사하고 원하는 값만 수정한다.

```bash
# YOLO 계열 실험
cp configs/yolo.yaml configs/exp_001_yolo11s.yaml

# RT-DETR 계열 실험
cp configs/rtdetr.yaml configs/exp_002_rtdetr_x.yaml
```

수정 예시 (`exp_001_yolo11s.yaml`):
```yaml
experiment:
  name: "exp_001_yolo11s"
  description: "YOLOv11s 모델, mosaic off"

model:
  architecture: "yolo11s.pt"

augmentation:
  mosaic: 0.0
```

수정 예시 (`exp_002_rtdetr_x.yaml`):
```yaml
experiment:
  name: "exp_002_rtdetr_x"
  description: "RT-DETR-X 백본"

model:
  architecture: "rtdetr-x.pt"
```

실행:
```bash
python src/train.py --config configs/exp_001_yolo11s.yaml
python src/train.py --config configs/exp_002_rtdetr_x.yaml
```

### 실험 비교 (MLflow)

MLflow UI에서 실험별 mAP50, precision, recall을 그래프로 비교 가능.

### 실험 비교 (DVC)

```bash
dvc params diff                 # 파라미터 변경점 확인
dvc metrics diff                # 메트릭 변경점 확인
```

---

## 주요 설정값

### YOLOv11 (configs/yolo.yaml)

```yaml
model:
  architecture: "yolo11n.pt"
train:
  epochs: 100
  batch_size: 16
  lr0: 0.001
  patience: 20
augmentation:
  mosaic: 1.0
inference:
  conf_threshold: 0.5
  iou_threshold: 0.45
```

### RT-DETR (configs/rtdetr.yaml)

```yaml
model:
  architecture: "rtdetr-l.pt"
data:
  image_size: 1280               # RT-DETR는 고해상도 권장
train:
  epochs: 1
  batch_size: 4                  # VRAM 절약
  lr0: 0.0001                    # Transformer는 낮은 lr
  warmup_epochs: 5               # 긴 warmup
  patience: 20
augmentation:
  mosaic: 0.0                    # RT-DETR에서는 off 권장
inference:
  conf_threshold: 0.5
  max_det: 300                   # Transformer query 수
```

전체 설정은 [configs/yolo.yaml](configs/yolo.yaml), [configs/rtdetr.yaml](configs/rtdetr.yaml) 참조.

---

## 서비스 연동

`src/utils.py`는 운영 코드와 공유 가능하도록 설계되었다.
YOLO, RT-DETR 모두 Ultralytics Results 객체를 반환하므로 동일한 후처리 코드를 사용한다.

```python
# 운영 코드에서 import 예시
from src.utils import parse_yolo_results, filter_by_class, compute_proximity

# 추론 결과 → 구조화 (YOLO, RT-DETR 모두 동일)
frame_result = parse_yolo_results(results[0], class_names=["person", "helmet", "hoist"])

# 클래스별 필터링
hoists = filter_by_class(frame_result, ["hoist"])
persons = filter_by_class(frame_result, ["person", "helmet"])

# 근접도 판별
for person in persons:
    for hoist in hoists:
        if compute_proximity(person, hoist, proximity_ratio=0.3):
            print(f"근접 감지: person {person.bbox} ↔ hoist {hoist.bbox}")
```

---

## 기술 스택

| 구분 | 도구 | 역할 |
|------|------|------|
| 모델 | Ultralytics YOLOv11 / RT-DETR | 객체 탐지 학습/추론 |
| 프레임워크 | PyTorch | 딥러닝 백엔드 |
| 데이터 버전 관리 | DVC | 이미지/라벨 버전 추적, 파이프라인 |
| 실험 추적 | MLflow | 하이퍼파라미터, 메트릭, 모델 아티팩트 |
| HP 튜닝 | Optuna | 하이퍼파라미터 자동 탐색 (TPE 알고리즘) |
| 설정 관리 | YAML | 실험별 config 분리 |

# vision-train-framework
