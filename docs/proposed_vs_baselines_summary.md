# Proposed Model vs Baselines Summary

작성 목적: 제안모델과 비교모델들이 각각 무엇을 학습했고, 무엇으로 평가했는지, 현재 남아있는 결과가 어떤 의미인지 정리한다.

## 1. 전체 비교 구조

| 구분 | 모델/폴더 | 방법론 | 학습 조건 | 평가 조건 | 현재 결과 상태 |
|---|---|---|---|---|---|
| Proposed | `entro2.py` + `test_ver7.py` | weather-driven forecaster + residual detector | `entro2`: clean 정상 발전 예측기 학습, `test_ver7`: SA 5/10 detector 학습 | `dataset6.0_attack` SA 5/8/10 | 정식 결과 있음 |
| Zhang baseline | `ai/final_experiment_package` | CNN-LSTM forecast + prediction interval anomaly | clean 기반 예측구간 모델 | 자체 clean test period에 SA 5/8/10 합성 | 정식 재현 결과 있음, test7 split과는 다름 |
| Zhang aligned | `zhang_test7_aligned` | Zhang-style CNN-LSTM + PI 유지 | `dataset5.4_attack` clean train/calibration | `dataset6.0_attack` SA 5/8/10 | 구현 완료, 현재 결과는 smoke |
| Tufail/compare2 aligned | `compare2_test7_aligned` | RF/MLP/CNN-LSTM/soft-voting supervised ensemble | `dataset5.4_attack` clean + SA 5/10 | `dataset6.0_attack` SA 5/8/10 | 구현 완료, 현재 결과는 smoke |
| Sensor/weather supervised ablation | `paper_weather_only_compare` | supervised RF/HGB/MLP ensemble | `dataset5.4_attack` clean + SA 5/10 | `dataset6.0_attack` SA 5/8/10 | 정식 실행 결과 있음 |

핵심 해석:

- Zhang baseline은 논문 방법론상 attack label을 학습하지 않는 clean-only anomaly detector다.
- Tufail/compare2 aligned는 제안모델 detector와 같이 attack label을 학습하는 supervised baseline이다.
- 제안모델은 clean forecaster와 supervised residual detector를 결합한다.

## 2. Proposed Model

### 2.1 Forecaster: `entro2.py`

역할:

- 정상 조건에서의 `power_ratio`를 예측한다.
- 발전량 lag는 쓰지 않고 weather/time/GHI dynamic feature를 사용한다.
- detector가 사용할 residual의 기준값을 만든다.

학습/설정:

- 입력 데이터: `dataset_clean/*2016_2019_clean.csv`
- target: `power_ratio`
- 주요 feature:
  - `ghi`, `temp`
  - `sin_hour`, `cos_hour`, `sin_doy`, `cos_doy`
  - `ghi_lag1,2,3,4,5,6,12,24`
  - `dghi_*`, `abs_dghi_*`, `slope_ghi_*`
  - `ghi_roll_mean/std/range`, `abs_dghi_1_roll_mean`
  - `capacity_kw`
- residual centering:
  - clean calibration last 60 days
  - GHI-bin별 residual bias 보정

Forecaster 결과:

| Model | Split | R2 | MAE | RMSE |
|---|---:|---:|---:|---:|
| LightGBM calibrated | Train | 0.9522 | 2.595% | 4.523% |
| LightGBM calibrated | Test | 0.9702 | 2.088% | 3.858% |
| CatBoost calibrated | Train | 0.9556 | 2.469% | 4.361% |
| CatBoost calibrated | Test | 0.9708 | 2.013% | 3.816% |

출처:

- `model_output_robust/metrics_compare.txt`
- `model_output_robust/fixed_forecaster_config.json`

### 2.2 Detector: `test_ver7.py`

역할:

- `entro2` forecaster 예측값과 실제 `power_ratio`로 residual을 만든다.
- residual window feature와 GHI dynamics feature를 사용해 attack detector를 학습한다.

학습 조건:

- train clean: `dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv`
- train attack:
  - `site5_5.4kw_2018_2020_attack_sa_5pct.csv`
  - `site5_5.4kw_2018_2020_attack_sa_10pct.csv`
- validation:
  - train feature 전체를 날짜 기준 80/20 split
- label:
  - `attack_label > 0.2`
- GHI filter:
  - `ghi >= 200`
- final threshold:
  - mid zone: 0.45
  - high zone: 0.55

평가 조건:

- `dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_5pct.csv`
- `dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_8pct.csv`
- `dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_10pct.csv`

결과:

| Dataset | Precision | Recall | F1 | FPR | Day Recall | Day FPR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.8271 | 0.9460 | 0.8825 | 0.0099 | 0.8980 | 0.2188 |
| SA 5% | 0.8737 | 0.6107 | 0.7189 | 0.0052 | 0.6684 | 0.1740 |
| SA 8% | 0.9024 | 0.8315 | 0.8655 | 0.0053 | 0.8421 | 0.1740 |
| SA 10% | 0.9157 | 0.9125 | 0.9141 | 0.0049 | 0.9053 | 0.1720 |

출처:

- `detector_model_fadre_v3/metrics_iter.json`

## 3. Zhang Baseline in `ai/`

위치:

- script: `ai/final_experiment_package/scripts/paper_cnn_lstm_baseline.py`
- result: `ai/final_experiment_package/results/paper_cnn_lstm_30min_metrics_summary.csv`

방법론:

- Zhang-style CNN-LSTM deterministic forecast
- Gaussian/prediction interval 기반 anomaly detection
- clean으로 정상 발전 예측기를 학습한다.
- calibration residual로 prediction interval을 만든다.
- 실제값이 interval 밖이면 anomaly로 판정한다.

입력 feature:

- `power_ratio`
- `ghi`
- `temp`

학습/평가 조건:

- clean 4-site PV data를 사용한다.
- clean test period에 SA 5/8/10 attack을 합성한다.
- 이 결과는 Zhang 방법론 재현용으로는 적절하지만, `test_ver7`과 동일 split/동일 attack CSV 평가는 아니다.

결과:

| Scenario | Precision | Recall | FPR | F1 | Day Recall | Day FPR |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.0000 | 0.0000 | 0.4166 | 0.0000 | 0.0000 | 0.9835 |
| SA 5% | 0.0678 | 0.5940 | 0.4171 | 0.1216 | 1.0000 | 0.9655 |
| SA 8% | 0.0852 | 0.7617 | 0.4176 | 0.1532 | 1.0000 | 0.9655 |
| SA 10% | 0.0903 | 0.8121 | 0.4174 | 0.1626 | 1.0000 | 0.9655 |

해석:

- 공격 recall은 높게 나오지만 FPR이 매우 높다.
- 기존 결과는 “Zhang 방법론 재현 baseline”으로 쓰고, 동일 `test_ver7` 조건 비교는 아래 `zhang_test7_aligned`를 사용한다.

## 4. Zhang Test7-Aligned

위치:

- script: `zhang_test7_aligned/scripts/zhang_test7_aligned.py`
- result: `zhang_test7_aligned/results/metrics_summary.csv`

방법론:

- Zhang baseline과 동일하게 clean-only CNN-LSTM prediction interval anomaly detector다.
- 공격 label은 학습에 쓰지 않고 평가에만 사용한다.

학습 조건:

- train/calibration clean:
  - `dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv`
- split:
  - 날짜 기준 80/20
- feature:
  - `power_ratio`, `ghi`, `temp`
- target:
  - future/current `power_ratio` sequence target
- GHI filter:
  - `ghi >= 200`

평가 조건:

- `dataset6.0_attack` SA 5/8/10
- label:
  - `attack_label > 0.2`

현재 결과 상태:

- 현재 저장된 결과는 `--epochs 1 --max-train-samples 5000 --max-eval-samples 3000` smoke test다.
- 정식 비교 수치로 사용하면 안 된다.

Smoke 결과:

| Dataset | Precision | Recall | F1 | FPR | Day Recall | Day FPR |
|---|---:|---:|---:|---:|---:|---:|
| SA 5% | 0.0330 | 0.1754 | 0.0556 | 0.3107 | 0.6633 | 0.6788 |
| SA 8% | 0.0309 | 0.1637 | 0.0519 | 0.3107 | 0.6633 | 0.6788 |
| SA 10% | 0.0287 | 0.1520 | 0.0483 | 0.3107 | 0.6531 | 0.6788 |

정식 실행:

```bash
cd /home/inseok/workspace/green_hyp/zhang_test7_aligned
conda run -n green_hy python scripts/zhang_test7_aligned.py
```

## 5. Tufail/Compare2 Test7-Aligned

위치:

- original method script: `compare2/compare2.py`
- aligned script: `compare2_test7_aligned/scripts/compare2_test7_aligned.py`
- result: `compare2_test7_aligned/results/comparison_summary.csv`

방법론:

- Tufail-style supervised ensemble classifier
- RF
- MLP
- CNN-LSTM
- soft-voting ensembles
- 기존 4-class `N/R/S/T` 방법론은 test7 조건에 맞추기 위해 binary attack detection으로 변환했다.

학습 조건:

- train clean:
  - `dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv`
- train attack:
  - `dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_5pct.csv`
  - `dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_10pct.csv`
- validation:
  - 날짜 기준 80/20
- label:
  - `attack_label > 0.2`
- feature mode currently checked:
  - `paper_with_time`: `ghi`, `power_ratio`, `temp`, `sin_hour`, `cos_hour`
- GHI filter:
  - `ghi >= 200`

평가 조건:

- `dataset6.0_attack` SA 5/8/10

현재 결과 상태:

- 현재 저장된 결과는 `--epochs 1 --max-train-windows 1000 --max-eval-windows 500` smoke test다.
- 정식 비교 수치로 사용하면 안 된다.

Smoke 결과:

| Dataset | Precision | Recall | F1 | FPR | Day Recall | Day FPR |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.3333 | 0.2778 | 0.3030 | 0.0207 | 0.3333 | 0.0533 |
| SA 5% | 0.1034 | 0.1250 | 0.1132 | 0.0546 | 0.1667 | 0.0734 |
| SA 8% | 0.1333 | 0.1667 | 0.1481 | 0.0546 | 0.2083 | 0.0734 |
| SA 10% | 0.1613 | 0.2083 | 0.1818 | 0.0546 | 0.2500 | 0.0734 |

정식 실행:

```bash
cd /home/inseok/workspace/green_hyp/compare2_test7_aligned
conda run -n green_hy python scripts/compare2_test7_aligned.py --feature-mode paper_with_time
```

전체 feature mode 실행:

```bash
cd /home/inseok/workspace/green_hyp/compare2_test7_aligned
conda run -n green_hy python scripts/compare2_test7_aligned.py --feature-mode all
```

## 6. Sensor/Weather Supervised Ablation

위치:

- script: `paper_weather_only_compare/paper_weather_only_compare.py`
- result: `paper_weather_only_compare/output/comparison_summary.csv`

역할:

- 논문 baseline이라기보다는 입력 feature 조건을 바꿔 직접 supervised classifier가 어느 정도 되는지 보는 ablation이다.
- `test_ver7`과 같은 train/eval attack CSV 조건을 사용한다.

학습 조건:

- train clean:
  - `dataset5.4_attack/site5_5.4kw_2018_2020_clean.csv`
- train attack:
  - SA 5%
  - SA 10%
- validation:
  - 날짜 기준 80/20

평가 조건:

- `dataset6.0_attack` SA 5/8/10

결과:

| Feature Mode | Dataset | Precision | Recall | F1 | FPR | Day Recall | Day FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| weather_only | Validation | 0.0632 | 0.1972 | 0.0957 | 0.0868 | 0.6939 | 0.6087 |
| weather_only | SA 5% | 0.0792 | 0.1728 | 0.1087 | 0.1063 | 0.7211 | 0.6839 |
| weather_only | SA 8% | 0.0792 | 0.1728 | 0.1087 | 0.1063 | 0.7211 | 0.6839 |
| weather_only | SA 10% | 0.0792 | 0.1728 | 0.1087 | 0.1063 | 0.7211 | 0.6839 |
| weather_power_ratio | Validation | 0.9202 | 0.7622 | 0.8338 | 0.0020 | 0.9388 | 0.0932 |
| weather_power_ratio | SA 5% | 0.1679 | 0.3873 | 0.2342 | 0.1016 | 0.6053 | 0.4553 |
| weather_power_ratio | SA 8% | 0.2223 | 0.5486 | 0.3164 | 0.1016 | 0.7684 | 0.4553 |
| weather_power_ratio | SA 10% | 0.2546 | 0.6557 | 0.3668 | 0.1016 | 0.8684 | 0.4553 |

해석:

- `power_ratio`를 넣으면 validation 성능은 크게 오른다.
- 하지만 external `dataset6.0_attack` 평가에서는 FPR이 높고 F1이 제안모델보다 낮다.
- 이는 direct sensor classifier보다 residual dynamics detector가 external system에서 더 안정적이라는 보조 근거가 된다.

## 7. 현재 결론

### 정식 비교에 바로 쓸 수 있는 결과

| Model | SA5 F1 | SA8 F1 | SA10 F1 | 비고 |
|---|---:|---:|---:|---|
| Proposed `entro2 + test_ver7` | 0.7189 | 0.8655 | 0.9141 | 정식 결과 |
| Zhang original reproduction in `ai` | 0.1216 | 0.1532 | 0.1626 | 정식 재현 결과, same split 아님 |
| supervised weather_power_ratio ablation | 0.2342 | 0.3164 | 0.3668 | same attack CSV 조건 |
| supervised weather_only ablation | 0.1087 | 0.1087 | 0.1087 | same attack CSV 조건 |

### 아직 full run이 필요한 결과

| Model | 현재 상태 | 정식 실행 명령 |
|---|---|---|
| Zhang test7-aligned | smoke only | `conda run -n green_hy python scripts/zhang_test7_aligned.py` |
| Tufail/compare2 test7-aligned | smoke only | `conda run -n green_hy python scripts/compare2_test7_aligned.py --feature-mode paper_with_time` |

## 8. 한글 해석 및 보고서용 문장

### 8.1 전체 해석

본 실험의 핵심은 비교모델을 하나만 보는 것이 아니라, 비교모델의 학습 방식에 따라 두 가지 관점으로 나누어 보는 것이다.

첫째, Zhang-style CNN-LSTM prediction interval baseline은 clean 데이터만으로 정상 발전량 예측 모델을 학습하고, clean calibration residual로 prediction interval을 만든 뒤, 공격 데이터에서는 실제 발전량이 예측구간 밖으로 벗어나는지를 anomaly로 판정한다. 이 방식은 공격 label을 학습에 사용하지 않는 clean-only anomaly detection 방식이다. 따라서 제안모델처럼 SA 5/10 공격 label을 detector 학습에 사용하는 방식과 학습 조건이 완전히 같지는 않다. 그러나 Zhang 논문 방법론 자체를 보존한 baseline으로는 적절하다.

둘째, Tufail/compare2 aligned baseline은 RF, MLP, CNN-LSTM, soft-voting ensemble이라는 supervised classification 방법론을 유지하면서, 제안모델과 동일하게 `dataset5.4_attack`의 clean + SA 5/10으로 학습하고 `dataset6.0_attack` SA 5/8/10으로 평가하도록 맞춘 비교모델이다. 이 모델은 공격 label을 학습에 사용하므로 제안모델의 detector 학습 조건과 더 가깝다.

셋째, 제안모델은 clean 데이터로 정상 발전량 forecaster를 먼저 학습하고, 그 예측값과 실제 발전량의 차이에서 나온 residual dynamics를 이용해 SA attack detector를 학습한다. 즉 단순히 센서값을 직접 분류하는 것이 아니라, 기상 조건에서 기대되는 정상 발전량과 실제 발전량 사이의 불일치를 공격 탐지의 핵심 신호로 사용한다.

현재 정식 결과 기준으로 보면, 제안모델은 SA 5/8/10에서 각각 F1 `0.7189`, `0.8655`, `0.9141`을 보였고 point-level FPR은 약 `0.5%` 수준이다. 반면 Zhang original baseline은 F1이 `0.1216`, `0.1532`, `0.1626` 수준이고 FPR이 약 `41.7%`로 높다. 또한 직접 supervised sensor classifier 계열 ablation에서도 external 평가에서는 F1과 FPR 균형이 제안모델보다 낮게 나타났다. 따라서 결과 해석은 “단순 prediction interval 또는 직접 센서 분류보다, 기상 기반 정상 발전량 예측 residual을 사용하는 방식이 SA 공격에서 더 안정적이다”로 정리할 수 있다.

### 8.2 모델별 해석

#### Proposed `entro2 + test_ver7`

제안모델은 두 단계 구조다. 먼저 `entro2`가 clean 데이터로 정상 발전량을 예측하는 forecaster를 학습한다. 이때 발전량 lag를 쓰지 않고 `ghi`, `temp`, 시간 feature, GHI lag/dynamic feature를 사용하므로 공격 발전량을 따라가는 leakage를 줄인다. 이후 `test_ver7`은 이 forecaster의 예측값과 실제 `power_ratio` 사이의 residual을 만들고, residual의 이동통계, 추세, 변동성, GHI coherence 등을 이용해 SA attack을 탐지한다.

이 구조의 장점은 공격이 발전량에 섞여 들어와도, 모델이 “기상 조건상 정상적으로 나와야 할 발전량”을 기준으로 차이를 보기 때문에 direct sensor classifier보다 attack signal을 더 안정적으로 포착할 수 있다는 점이다.

#### Zhang baseline

Zhang baseline은 clean-only prediction interval 방식이다. clean 데이터로 CNN-LSTM forecaster를 학습하고, clean calibration residual로 예측구간을 만든다. 이후 attack 데이터에서 실제 발전량이 예측구간을 벗어나면 anomaly로 본다.

이 방식은 공격 label을 학습에 쓰지 않기 때문에 실사용 관점에서는 장점이 있지만, 예측구간이 넓거나 정상 변동성이 큰 경우 false alarm이 많아질 수 있다. 현재 기존 `ai` 결과에서도 FPR이 약 `41.7%`로 높게 나타났고, 이는 prediction interval 방식이 SA 공격을 어느 정도 잡더라도 정상 구간 오탐이 많을 수 있음을 보여준다.

#### Tufail/compare2 aligned baseline

Tufail/compare2 aligned baseline은 supervised ensemble classifier다. RF, MLP, CNN-LSTM 및 soft-voting ensemble 구조를 유지하되, 실험 조건은 제안모델과 맞췄다. 즉 `dataset5.4_attack` clean + SA 5/10으로 학습하고 `dataset6.0_attack` SA 5/8/10으로 평가한다.

이 baseline은 “공격 label을 주고 직접 센서 feature를 분류하면 어느 정도 되는가”를 보는 비교모델이다. 제안모델과 같은 공격 학습 조건을 공유하므로 supervised 비교로 의미가 있다. 다만 방법론상 residual representation을 쓰지 않고 센서값 자체를 분류하기 때문에, 외부 시스템으로 넘어갈 때 발전소 규모나 분포 차이에 취약할 수 있다.

#### Weather-only / weather+power_ratio ablation

`weather_only`는 `ghi`, `temp`, 시간/GHI dynamic feature만 직접 classifier에 넣은 실험이다. SA 공격은 주로 발전량 쪽을 조작하므로, 기상값만으로 공격 label을 직접 분리하기 어렵다. 실제 결과도 F1이 약 `0.1087` 수준으로 낮았다. 이는 “기상 feature만 직접 분류기로 넣는 방식은 부족하며, 기상 조건에서 기대되는 정상 발전량을 먼저 예측한 뒤 residual을 보는 구조가 필요하다”는 근거가 된다.

`weather_power_ratio`는 현재 발전량 비율인 `power_ratio`를 추가한 supervised ablation이다. validation에서는 성능이 크게 오르지만 external `dataset6.0_attack` 평가에서는 F1이 SA 10%에서도 `0.3668`에 머문다. 즉 발전량 센서를 직접 보더라도, 단순 supervised classifier는 제안모델의 residual dynamics detector만큼 일반화되지 않는다는 해석이 가능하다.

### 8.3 보고서용 문장

제안모델 설명:

> 제안모델은 먼저 기상 조건에 기반한 정상 태양광 발전량을 예측하고, 이후 실제 발전량과 예측 발전량 사이의 residual dynamics를 이용하여 SA 공격을 탐지한다. 이를 통해 단순 센서값 분류가 아니라, 기상 조건상 기대되는 정상 발전 패턴으로부터의 이탈을 공격 신호로 활용한다.

Zhang baseline 설명:

> Zhang-style baseline은 clean 데이터만으로 CNN-LSTM 발전량 예측 모델을 학습하고, clean calibration residual로 prediction interval을 설정한 뒤, 실제 발전량이 예측구간을 벗어나는 경우 anomaly로 판정하는 방식이다. 이 baseline은 공격 label을 학습에 사용하지 않는 clean-only anomaly detection 비교모델이다.

Tufail/compare2 aligned baseline 설명:

> Tufail-style baseline은 RF, MLP, CNN-LSTM 및 soft-voting ensemble을 사용하는 supervised classification 방법론을 유지하되, 제안모델과 동일한 SA 학습/평가 조건에서 비교하기 위해 `dataset5.4_attack`의 clean + SA 5/10으로 학습하고 `dataset6.0_attack`의 SA 5/8/10으로 평가하였다.

결과 해석 문장:

> 실험 결과, clean-only prediction interval 방식과 직접 supervised sensor classifier는 external SA 평가에서 높은 오탐률 또는 낮은 F1을 보였다. 반면 제안모델은 SA 5/8/10에서 각각 F1 0.7189, 0.8655, 0.9141을 달성하고 point-level FPR을 약 0.5% 수준으로 유지하였다. 이는 기상 기반 정상 발전량 예측과 residual dynamics를 결합한 구조가 SA 공격 탐지에 더 효과적임을 보여준다.

### 8.4 영문 보고서용 문장

제안모델:

> The proposed method first learns weather-conditioned normal PV generation using a robust forecaster and then trains a residual-dynamics detector using labeled SA attacks.

Zhang baseline:

> The Zhang-style baseline follows a clean-trained CNN-LSTM prediction-interval anomaly detection setting. It does not use labeled attacks for training; attack labels are used only for evaluation.

Tufail/compare2 aligned baseline:

> For a fair supervised comparison, the Tufail-style RF/MLP/CNN-LSTM ensemble was evaluated under the same train/evaluation SA split as the proposed detector, while preserving the original supervised ensemble methodology.

핵심 해석:

> The clean-only prediction-interval baseline and direct supervised sensor classifiers show limited robustness under external SA evaluation, whereas the proposed weather-conditioned residual detector achieves substantially higher F1 with much lower point-level FPR.
