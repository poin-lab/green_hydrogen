# Green Hydrogen FDI Anomaly Detection

그린수소 생산 및 재생에너지 데이터 관리 환경에서 태양광 발전 데이터의 무결성을
검증하기 위한 AI 기반 이상 탐지 프로젝트입니다.

본 프로젝트는 재생에너지 발전량 데이터에 대한 False Data Injection, FDI 공격을
탐지하는 것을 목표로 합니다. 정상 발전 패턴을 먼저 학습한 뒤, 실제 발전량과 예측값의
차이인 residual을 기반으로 공격 여부를 판별합니다. 그린수소 생산량 산정, 인증, 거래,
정산 과정에서 발전 데이터가 조작될 경우 생길 수 있는 문제를 줄이기 위한 탐지 계층을
구현해보는 데 초점을 두었습니다.

## 핵심 아이디어

```text
날씨/시간 기반 정상 발전량 예측 -> residual 계산 -> FDI 공격 탐지
```

제안 모델은 두 단계로 구성됩니다.

1. 정상 발전량 예측기

   Clean 태양광 발전 데이터와 날씨, 시간, GHI 변화량 feature를 이용해 정상적인
   `power_ratio`를 예측합니다. 이 단계는 공격 label을 사용하지 않고 정상 발전 패턴만
   학습합니다.

2. Residual 기반 공격 탐지기

   예측 발전량과 실제 발전량의 차이를 window feature로 만들고, GHI 구간별 LightGBM
   detector를 학습해 FDI 공격 여부를 분류합니다.

## 정리 방향

기존 실험 폴더 `green_hyp`에는 raw dataset, feature cache, 모델 checkpoint, 중간 결과,
zip 파일 등이 함께 섞여 있어 용량이 약 13GB였습니다. 이 저장소용 폴더는 공개 및 제출에
필요한 핵심 코드, 문서, 요약 결과만 남긴 정리본입니다.

포함한 항목:

- 제안 모델 코드
- 비교 모델 코드
- 최종 요약 결과 CSV/JSON
- 데이터 사용 protocol 문서
- 실험 및 비교 모델 설명 문서

제외한 항목:

- Raw CSV dataset
- Feature cache
- 학습된 모델 파일
- 대용량 prediction CSV
- `__pycache__`
- zip 백업 파일
- 중간 실험 plot 및 임시 결과물

## 폴더 구조

```text
.
├── proposed/
│   ├── entro2.py
│   ├── proposed_entro2_test7_pipeline.py
│   ├── test_ver7.py
│   └── test_ver7_eval_slowramp.py
├── baselines/
│   ├── compare2_test7_aligned.py
│   ├── compare2_slowramp_eval.py
│   ├── feature_matched_compare.py
│   ├── zhang_entro_clean_only_aligned.py
│   ├── zhang_entro_clean_only_slowramp_eval.py
│   └── zhang_multi_clean_attack_tuned.py
├── results/
│   ├── final_strict_with_accuracy_summary.csv
│   ├── slowramp_main_comparison_summary.csv
│   └── summary metric files
├── docs/
│   └── experiment notes
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

## 사용 데이터

실제 raw dataset은 용량 문제로 저장소에 포함하지 않습니다. 실행 시에는 프로젝트 루트에
아래 폴더를 준비해야 합니다.

```text
dataset_clean/
dataset5.4_attack/
dataset6.0_attack/
```

자세한 파일 목록과 학습/평가 분할은 [data/README.md](data/README.md)를 참고하면 됩니다.

## 주요 결과

SA 공격에 대한 strict point-wise 기준 대표 결과입니다.

| Model | Attack | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | SA 5% | 0.9736 | 0.8737 | 0.6107 | 0.7189 | 0.0052 |
| Proposed | SA 8% | 0.9857 | 0.9024 | 0.8315 | 0.8655 | 0.0053 |
| Proposed | SA 10% | 0.9905 | 0.9157 | 0.9125 | 0.9141 | 0.0049 |

Slow-ramp 공격에 대한 대표 결과입니다.

| Model | Attack | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Proposed | Ramp 8% | 0.9793 | 0.8686 | 0.6786 | 0.7619 | 0.0053 |
| Proposed | Ramp 10% | 0.9879 | 0.8929 | 0.8555 | 0.8738 | 0.0053 |

전체 비교 결과는 `results/` 폴더의 CSV/JSON 파일에 정리되어 있습니다.

## 실행

의존성 설치:

```bash
pip install -r requirements.txt
```

제안 모델 전체 실행:

```bash
python proposed/proposed_entro2_test7_pipeline.py
```

Slow-ramp 평가:

```bash
python proposed/test_ver7_eval_slowramp.py
```

비교 모델 예시:

```bash
python baselines/zhang_entro_clean_only_aligned.py
python baselines/compare2_test7_aligned.py
python baselines/feature_matched_compare.py
```

## 기술 스택

- Python
- pandas, NumPy, SciPy
- scikit-learn
- LightGBM
- CatBoost
- PyTorch
- Matplotlib
- joblib

## 참고 문서

- [docs/proposed_vs_baselines_summary.md](docs/proposed_vs_baselines_summary.md)
- [docs/slowramp_results_ko.md](docs/slowramp_results_ko.md)
- [docs/cnn_lstm_clean_only_aligned_ko.md](docs/cnn_lstm_clean_only_aligned_ko.md)
- [docs/what_to_include_in_comparison_ko.md](docs/what_to_include_in_comparison_ko.md)
