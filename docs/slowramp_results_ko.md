# Slow-Ramp Attack 평가 결과

## 목적

SA 공격으로 학습한 모델들이 slow-ramp 공격에도 일반화되는지 확인했다. Ramp는 새로 학습에 넣지 않고, 기존 SA 학습 조건을 유지한 상태에서 external slow-ramp 파일만 평가했다.

## 데이터

학습 조건은 기존 main comparison과 동일하다.

| Model | 학습 조건 | Ramp 평가 |
|---|---|---|
| Proposed `entro2+test_ver7` | `dataset5.4_attack` SA5/SA10 detector 학습 | `dataset6.0_attack` slowramp 8/10 |
| CNN-LSTM PI single-site aligned | `dataset_clean` 4개 clean-only 학습, attack 학습 없음 | `dataset6.0_attack` slowramp 8/10 |
| Tufail paper_with_time | `dataset5.4_attack` clean + SA5/SA10 supervised 학습 | `dataset6.0_attack` slowramp 8/10 |

평가 파일:

- `dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_8pct.csv`
- `dataset6.0_attack/site5_6.0kw_2021_2022_attack_slowramp_10pct.csv`

## 실행 코드

| Model | 실행 코드 |
|---|---|
| Proposed | `final_model_comparison_code/proposed/test_ver7_eval_slowramp.py` |
| CNN-LSTM PI | `final_model_comparison_code/baselines/zhang_entro_clean_only_slowramp_eval.py` |
| Tufail | `final_model_comparison_code/baselines/compare2_slowramp_eval.py` |

## 결과

요약 CSV:

- `final_model_comparison_code/results/slowramp_main_comparison_summary.csv`

| Model | Method type | Attack | Accuracy | Precision | Recall | F1 | FPR | Day Recall | Day FPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed `entro2+test_ver7` | Two-stage LightGBM residual detector | Ramp8 | 0.9793 | 0.8686 | 0.6786 | 0.7619 | 0.0053 | 0.3143 | 0.1679 |
| Proposed `entro2+test_ver7` | Two-stage LightGBM residual detector | Ramp10 | 0.9879 | 0.8929 | 0.8555 | 0.8738 | 0.0053 | 0.3714 | 0.1679 |
| CNN-LSTM PI single-site aligned | Deep learning PI clean-only | Ramp8 | 0.5885 | 0.0618 | 0.5089 | 0.1103 | 0.4073 | 0.8857 | 0.9649 |
| CNN-LSTM PI single-site aligned | Deep learning PI clean-only | Ramp10 | 0.5889 | 0.0627 | 0.5167 | 0.1119 | 0.4073 | 0.9143 | 0.9649 |
| Tufail paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | Ramp8 | 0.9108 | 0.1442 | 0.1581 | 0.1508 | 0.0495 | 0.6000 | 0.3649 |
| Tufail paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | Ramp10 | 0.9120 | 0.1635 | 0.1835 | 0.1729 | 0.0495 | 0.6286 | 0.3649 |

평균 요약:

| Model | Avg F1 | Avg FPR |
|---|---:|---:|
| Proposed `entro2+test_ver7` | 0.8179 | 0.0053 |
| CNN-LSTM PI single-site aligned | 0.1111 | 0.4073 |
| Tufail paper_with_time | 0.1619 | 0.0495 |

## 해석

Proposed는 SA5/SA10으로 detector를 학습했지만 slow-ramp 8/10 평가에서도 F1 `0.7619`, `0.8738`을 보였다. FPR도 `0.0053`으로 SA 평가와 비슷하게 낮게 유지된다.

CNN-LSTM PI는 Ramp에서도 기존 SA 평가와 비슷하게 FPR이 매우 높다. Day Recall은 높지만 Day FPR도 `0.9649`라 정상일 대부분을 이상으로 판단한다.

Tufail supervised ensemble은 FPR은 `0.0495`로 CNN-LSTM PI보다 낮지만, Ramp recall이 낮아 F1이 `0.1508`, `0.1729`에 그친다. 이는 SA 학습으로 얻은 직접분류 decision boundary가 slow-ramp attack으로 충분히 일반화되지 못했음을 시사한다.

## 논문/보고서용 문장

영문:

> To evaluate generalization to a different attack pattern, we additionally tested all models on external slow-ramp attacks without adding slow-ramp samples to the training data. The proposed two-stage LightGBM residual detector maintained high F1 scores of 0.7619 and 0.8738 for Ramp 8% and Ramp 10%, respectively, while keeping the FPR at 0.0053. In contrast, the CNN-LSTM prediction-interval baseline suffered from high false alarms, and the Tufail-style supervised ensemble showed limited recall on slow-ramp attacks.

국문:

> 서로 다른 공격 패턴에 대한 일반화 성능을 확인하기 위해 slow-ramp 공격을 학습에 추가하지 않고 외부 slow-ramp 8/10% 데이터에서 추가 평가하였다. 제안모델은 Ramp8/Ramp10에서 각각 F1 `0.7619`, `0.8738`을 달성하면서 FPR을 `0.0053`으로 낮게 유지하였다. 반면 CNN-LSTM PI baseline은 높은 오탐률을 보였고, Tufail-style supervised ensemble은 slow-ramp 공격에 대한 recall이 낮아 제한적인 F1을 보였다.
