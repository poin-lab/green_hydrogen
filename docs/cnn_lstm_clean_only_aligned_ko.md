# CNN-LSTM Prediction Interval Baseline 정합성 및 결과

## 목적

팀원이 제안한 CNN-LSTM baseline 정합성 원칙을 확인하고, 제안모델 protocol에 맞춘 clean-only CNN-LSTM prediction interval 비교모델을 추가했다.

핵심 원칙은 다음과 같다.

> CNN-LSTM prediction interval baseline은 clean 데이터로 정상 발전량 예측 모델을 학습하고, clean calibration residual로 prediction interval을 만든 뒤, 외부 공격 데이터에서 실제 발전량이 interval 밖으로 벗어나는지를 anomaly로 판정한다.

따라서 이 baseline은 공격 라벨을 학습에 사용하는 supervised detector가 아니다. 공격 라벨은 최종 평가 지표 계산에만 사용한다.

## 추가한 코드

| 항목 | 경로 |
|---|---|
| 실행 코드 | `final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py` |
| 결과 폴더 | `final_model_comparison_code/results/zhang_entro_clean_only_aligned/` |
| 요약 CSV | `final_model_comparison_code/results/cnn_lstm_clean_only_comparison_summary.csv` |

## 방법론 검토

기존 `ai/final_experiment_package/scripts/paper_cnn_lstm_baseline.py`는 Zhang et al. 2022 CNN-LSTM deterministic forecast + prediction interval 흐름을 재현한 코드다.

확인된 방법론 요소:

- CNN-LSTM forecaster
- 입력 feature: `power_ratio`, `ghi`, `temp`
- clean 데이터 기반 예측 모델 학습
- clean residual 기반 prediction interval
- interval 밖이면 anomaly
- attack label은 원래 학습에 사용하지 않는 구조

하지만 기존 코드는 제안모델 비교용으로는 정합성이 맞지 않았다.

| 항목 | 기존 코드 | 정합 protocol |
|---|---|---|
| clean 학습 | 4-site clean | 유지 |
| attack 학습 | 없음 | `N/A`로 명시 |
| 평가 | clean test 구간에 synthetic attack 생성 | `dataset6.0_attack` CSV 직접 평가 |
| label | synthetic mask | CSV의 `attack_label` |
| spatial 구조 | 2x2 SPSM | 단일 6.0kW external site에는 mismatch |
| sigma 보정 | 추가 보정 있음 | clean residual quantile만 사용 |

## 왜 이렇게 수정했는가

제안모델 protocol은 다음과 같이 확정되어 있다.

| 단계 | 데이터 | 역할 |
|---|---|---|
| Stage 1 | `dataset_clean/site5_{5.9,7.0,226.8,327.6}kw_2016_2019_clean.csv` | 정상 발전량 forecaster 학습 |
| Stage 2 학습 | `dataset5.4_attack/site5_5.4kw_2018_2020_attack_sa_{5,10}pct.csv` | residual detector 학습 |
| Stage 2 평가 | `dataset6.0_attack/site5_6.0kw_2021_2022_attack_sa_{5,8,10}pct.csv` | external 평가 |

CNN-LSTM PI baseline은 attack label을 학습하지 않는 clean-trained anomaly detector이므로 Stage 2 학습은 `N/A`다. 이 방법론에 억지로 attack 학습을 넣으면 논문 방식이 아니라 attack-tuned ablation이 된다.

그래서 정합 비교에서는 다음처럼 구성했다.

| 단계 | CNN-LSTM PI aligned 설정 |
|---|---|
| Stage 1 clean 학습 | `entro2`와 동일한 `dataset_clean` 4개 CSV |
| Clean fit/calibration split | 날짜 기준 앞 80% fit, 뒤 20% clean interval calibration |
| Stage 2 attack 학습 | 없음 |
| Stage 3 평가 | `dataset6.0_attack` SA5/SA8/SA10 직접 평가 |
| 평가 label | CSV의 `attack_label > 0.2` |
| 입력 feature | `power_ratio`, `ghi`, `temp` |
| 모델 구조 | single-site CNN-LSTM-w/o-spatial 형태 |
| PI threshold | clean calibration residual의 70% quantile |

단일 6.0kW site 평가에서는 2x2 spatial matrix를 구성할 동시간대 4-site external attack 데이터가 없다. 따라서 spatial full model을 억지로 유지하지 않고, 논문 방법론의 핵심인 CNN-LSTM forecasting + prediction interval 판정은 유지하면서 single-site/w-o-spatial 형태로 맞췄다.

## 실행 조건

```bash
cd /home/inseok/workspace/green_hyp
conda run -n green_hy python final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py --epochs 5 --batch-size 4096 --threshold-source clean_quantile
```

학습 로그 요약:

| 항목 | 값 |
|---|---:|
| Device | CPU |
| Train sequences | 322,302 |
| Clean calibration sequences | 101,228 |
| Epochs | 5 |
| Batch size | 4096 |
| PI half-width | 0.01958 |
| Threshold source | clean calibration quantile |
| Attack train files | 없음 |

## 결과

| Dataset | Accuracy | Precision | Recall | F1 | FPR | PA-F1 | Day Recall | Day FPR | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SA5 | 0.5789 | 0.0649 | 0.4893 | 0.1146 | 0.4159 | 0.2210 | 1.0000 | 0.9600 | 0.0354 | 0.0680 |
| SA8 | 0.5807 | 0.0723 | 0.5519 | 0.1279 | 0.4176 | 0.2203 | 1.0000 | 0.9600 | 0.0357 | 0.0683 |
| SA10 | 0.5820 | 0.0766 | 0.5883 | 0.1355 | 0.4184 | 0.2199 | 1.0000 | 0.9600 | 0.0359 | 0.0685 |

Proposed와 strict point-level F1/FPR 비교:

| Model | SA5 F1 | SA8 F1 | SA10 F1 | SA5 FPR | SA8 FPR | SA10 FPR |
|---|---:|---:|---:|---:|---:|---:|
| Proposed `entro2+test_ver7` | 0.7189 | 0.8655 | 0.9141 | 0.0052 | 0.0053 | 0.0049 |
| CNN-LSTM PI entro-clean-only | 0.1146 | 0.1279 | 0.1355 | 0.4159 | 0.4176 | 0.4184 |

## Confidence Sweep

70% prediction interval 결과가 너무 낮아 보일 수 있으므로, interval confidence를 `0.70`, `0.80`, `0.90`, `0.95`, `0.99`로 바꿔 sweep을 수행했다. 목적은 CNN-LSTM PI baseline이 단순히 interval 폭 선택 때문에 불리해진 것인지 확인하는 것이다.

실행 조건:

```bash
cd /home/inseok/workspace/green_hyp
conda run -n green_hy python final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py --epochs 5 --batch-size 4096 --threshold-source clean_quantile --confidence 0.80
conda run -n green_hy python final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py --epochs 5 --batch-size 4096 --threshold-source clean_quantile --confidence 0.90
conda run -n green_hy python final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py --epochs 5 --batch-size 4096 --threshold-source clean_quantile --confidence 0.95
conda run -n green_hy python final_model_comparison_code/baselines/zhang_entro_clean_only_aligned.py --epochs 5 --batch-size 4096 --threshold-source clean_quantile --confidence 0.99
```

요약 파일:

- `final_model_comparison_code/results/cnn_lstm_pi_confidence_sweep_summary.csv`

| Confidence | Half-width | SA5 F1 | SA8 F1 | SA10 F1 | Avg F1 | Avg Recall | Avg FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.01958 | 0.1146 | 0.1279 | 0.1355 | 0.1260 | 0.5432 | 0.4173 |
| 0.80 | 0.02596 | 0.1161 | 0.1336 | 0.1481 | 0.1326 | 0.4139 | 0.2845 |
| 0.90 | 0.05265 | 0.0936 | 0.1064 | 0.1173 | 0.1058 | 0.2079 | 0.1605 |
| 0.95 | 0.10380 | 0.0763 | 0.0785 | 0.0821 | 0.0790 | 0.1015 | 0.0867 |
| 0.99 | 0.23136 | 0.0355 | 0.0397 | 0.0417 | 0.0390 | 0.0271 | 0.0215 |

해석:

- 0.70 PI는 recall이 가장 높지만 FPR이 약 `0.417`로 매우 높다.
- 0.80 PI에서 Avg F1이 `0.1326`으로 가장 높지만 여전히 proposed와 큰 차이가 있다.
- 0.90 이상으로 interval을 넓히면 FPR은 줄어들지만 recall이 급격히 감소한다.
- 0.99 PI는 FPR이 `0.0215`까지 내려가지만 Avg Recall이 `0.0271`로 공격을 거의 놓친다.

따라서 CNN-LSTM PI baseline의 낮은 성능은 단순히 70% confidence 선택 때문이 아니다. Prediction interval 방식은 external single-site SA 평가에서 interval 폭을 좁히면 오탐이 커지고, interval 폭을 넓히면 공격 탐지가 무너지는 trade-off를 보인다.

## 해석

CNN-LSTM PI baseline은 clean-only prediction interval 방식이라 attack pattern을 직접 학습하지 않는다. 이 때문에 외부 6.0kW 공격 평가에서 공격 recall은 어느 정도 확보되지만, clean 구간도 interval 밖으로 자주 벗어나면서 FPR이 약 `41.6%~41.8%`로 매우 높게 나타난다.

Day-level 기준으로도 Day Recall은 `1.0`이지만 Day FPR이 `0.96`이다. 즉 공격일은 거의 모두 잡지만, 정상일도 대부분 이상으로 울리는 구조다.

따라서 이 결과는 다음 주장을 뒷받침한다.

> Clean-trained CNN-LSTM prediction interval 방식은 외부 site/capacity/year 조건에서 정상 변동성과 공격 변동성을 충분히 분리하지 못해 높은 false alarm을 보인다. 제안모델은 clean forecaster를 사용하되 residual/window/GHI-zone feature와 supervised detector를 결합하여 F1과 FPR 모두에서 개선된다.

## 논문/보고서용 문장

영문:

> The CNN-LSTM prediction-interval baseline was implemented as a clean-trained unsupervised anomaly detector. It used the same multi-site clean data as the proposed forecaster for model training and clean residual calibration, and was evaluated directly on the external 6.0 kW SA attack datasets. No labeled attack data were used for training or threshold tuning. Under this aligned protocol, the baseline showed high recall but very high false positive rates, indicating that the prediction-interval decision rule alone is insufficient under unseen site/capacity/year conditions.

국문:

> CNN-LSTM prediction interval 비교모델은 clean 데이터만으로 학습되는 비지도 이상탐지 방식으로 구현하였다. 제안모델의 forecaster와 동일한 multi-site clean 데이터로 예측 모델과 clean residual 기반 interval을 학습하고, 외부 6.0kW SA 공격 데이터에서 직접 평가하였다. 공격 라벨 데이터는 학습 또는 threshold tuning에 사용하지 않았다. 정합 protocol에서 해당 baseline은 높은 recall을 보였으나 FPR이 매우 높게 나타났으며, 이는 prediction interval 판정만으로는 unseen site/capacity/year 조건의 정상 변동성과 공격 변동성을 충분히 분리하기 어렵다는 점을 보여준다.
