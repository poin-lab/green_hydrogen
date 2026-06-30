# 데이터 사용 설명

이 저장소에는 raw dataset을 포함하지 않습니다. 원본 CSV는 용량이 크고 실험 산출물과
함께 관리되기 때문에, GitHub에는 코드와 요약 결과만 올리는 것을 기준으로 합니다.

실험을 재실행하려면 프로젝트 루트에 아래 폴더를 준비합니다.

```text
green_hydrogen/
├── dataset_clean/
├── dataset5.4_attack/
└── dataset6.0_attack/
```

## 1. 정상 발전량 예측기 학습 데이터

`proposed/entro2.py`에서 사용합니다.

```text
dataset_clean/site5_5.9kw_2016_2019_clean.csv
dataset_clean/site5_7.0kw_2016_2019_clean.csv
dataset_clean/site5_226.8kw_2016_2019_clean.csv
dataset_clean/site5_327.6kw_2016_2019_clean.csv
```

용도:

- 정상 발전 패턴 학습
- LightGBM/CatBoost forecaster 학습
- residual calibration 기준 생성

## 2. Detector 학습 데이터

`proposed/test_ver7.py`와 supervised baseline에서 사용합니다.

```text
dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv
dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_5pct.csv
dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_10pct.csv
```

용도:

- residual 기반 detector 학습
- 날짜 기준 80:20 train/validation split
- SA 5%, SA 10% 공격 패턴 학습

## 3. External SA 평가 데이터

학습에 사용하지 않은 6.0kW, 2021-2022 데이터입니다.

```text
dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv
dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv
dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv
```

용도:

- 최종 SA 5%, SA 8%, SA 10% 일반화 성능 평가
- 제안 모델과 비교 모델의 동일 조건 비교

## 4. External slow-ramp 평가 데이터

```text
dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_8pct.csv
dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_10pct.csv
```

용도:

- 천천히 증가하는 ramp 공격에 대한 일반화 성능 평가

## 5. 주요 컬럼

스크립트에서 주로 사용하는 컬럼은 아래와 같습니다.

- `timestamp` 또는 시간 인덱스 계열 컬럼
- `power_ratio`
- `ghi`
- `temp`
- `attack_label`
- `capacity_kw`

일부 feature는 스크립트 내부에서 시간, GHI 변화량, rolling window 기반으로 생성합니다.

## 6. 저장소에 넣지 않는 데이터

아래 항목은 `.gitignore` 대상입니다.

- `dataset_clean/`
- `dataset5.4_attack/`
- `dataset6.0_attack/`
- `feature_cache*/`
- `model_output*/`
- `detector_model*/`
- `*.pkl`, `*.pt`, `*.cbm`, `*.joblib`
- 대용량 prediction CSV
