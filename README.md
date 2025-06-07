# Seeksick-KoBERT: 텍스트 기반 감정 분류 모델

멀티모달 감정 추론 시스템의 텍스트 분석 파트를 위한 KoBERT 기반 감정 분류 모델입니다.

## 환경 설정

### 1. micromamba 설치 (Windows)

```bash
# micromamba 설치
curl -L -O "https://micro.mamba.pm/api/micromamba/win-64/latest"
micromamba.exe create -n seeksick python=3.10
micromamba.exe activate seeksick
```

### 2. CUDA 12.6 설치
- [NVIDIA CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive) 다운로드 및 설치

### 3. 프로젝트 의존성 설치

```bash
pip install -r requirements.txt
```

## 프로젝트 구조

```
seeksick-kobert/
├── config/
│   └── config.yaml          # 설정 파일
├── data/
│   ├── raw/                 # 원본 데이터
│   └── processed/           # 전처리된 데이터
├── src/
│   ├── data/
│   │   ├── dataset.py       # 데이터셋 클래스
│   │   └── preprocessing.py # 데이터 전처리
│   ├── models/
│   │   └── kobert_classifier.py # KoBERT 모델
│   ├── utils/
│   │   ├── logger.py        # 로깅 유틸리티
│   │   └── metrics.py       # 평가 메트릭
│   └── train.py             # 학습 스크립트
└── requirements.txt         # 의존성 목록
```

## 데이터 준비

1. 원본 CSV 파일을 `data/raw/` 디렉토리에 위치
2. 전처리된 CSV 파일을 `data/processed/` 디렉토리에 위치
3. `config/config.yaml` 파일에서 데이터 경로 설정

## 모델 학습

```bash
python src/train.py
```

## 주요 기능

- KoBERT 기반 5차원 감정 분류 (행복, 우울, 놀람, 분노, 중립)
- 클래스 불균형 해결을 위한 가중치 적용
- 학습 과정 모니터링 (wandb)
- GPU 메모리 사용량 추적
- 자동 모델 체크포인트 저장

## 주의사항

1. CUDA 12.6과 호환되는 PyTorch 버전 사용
2. 데이터셋의 클래스 불균형 고려
3. GPU 메모리 사용량 모니터링
4. wandb 프로젝트 설정 필요