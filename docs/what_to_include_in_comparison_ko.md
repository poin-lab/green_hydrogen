# 비교 결과에 무엇을 넣을지 정리

## 1. 최종 추천 구성

비교 결과는 한 표에 전부 몰아넣지 말고 세 덩어리로 나누는 것이 가장 깔끔하다.

1. Main comparison
   - 논문 비교모델로 직접 비교할 모델만 넣는다.
   - Proposed, Zhang, Tufail을 넣는다.

2. Ablation study
   - 제안모델 구조가 왜 필요한지 설명하는 실험을 넣는다.
   - feature-matched baseline, weather-only, weather+power_ratio, Tufail engineered를 넣는다.

3. Appendix or supplementary
   - Zhang original reproduction처럼 split이 다른 결과를 넣는다.
   - 본문 main comparison에는 넣지 않는 것이 좋다.

## 2. Main Comparison에 넣을 모델

본문 메인 비교표에는 아래 네 개를 넣는 것을 추천한다.

| Model | 방법론 타입 | 넣는 이유 |
|---|---|---|
| Proposed `entro2 + test_ver7` | Two-stage ML: clean forecaster + residual-based supervised detector | 제안모델 |
| CNN-LSTM PI single-site aligned | Deep learning forecasting + prediction interval, clean-only anomaly detection | clean-only prediction interval 논문 baseline을 제안모델 protocol에 맞춘 버전 |
| Zhang attack-tuned | Deep learning forecasting + attack-tuned residual threshold | Zhang 구조에서 threshold만 공격 validation으로 맞춘 공정성 보강 baseline |
| Tufail/compare2 paper_with_time | Supervised ensemble learning: RF + MLP + CNN-LSTM soft voting | 공격 라벨 학습을 사용하는 supervised ensemble 논문 baseline |

`CNN-LSTM PI single-site aligned`와 `Zhang attack-tuned`를 둘 다 넣는 이유는 다음과 같다.

- CNN-LSTM PI single-site aligned는 attack label을 쓰지 않는 clean-only prediction interval 원칙을 보존한다.
- Zhang attack-tuned는 “비교모델도 공격 데이터로 threshold 맞춰야 하는 것 아니냐”는 지적에 대응한다.
- 둘 다 넣으면 Zhang에 불리하게 설정했다는 비판을 줄일 수 있다.

## 3. Main Comparison 결과표

| Model | 방법론 타입 | SA5 F1 | SA8 F1 | SA10 F1 | Avg F1 | Avg FPR | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Proposed | Two-stage residual ML detector | 0.7189 | 0.8655 | 0.9141 | 0.8328 | 0.0051 | 가장 높은 F1, 가장 낮은 FPR |
| CNN-LSTM PI single-site aligned | Deep learning PI, clean-only | 0.1146 | 0.1279 | 0.1355 | 0.1260 | 0.4173 | `entro2` clean 데이터만 사용한 clean-only PI, 오탐 매우 큼 |
| Zhang test7-aligned | Deep learning PI, clean-only | 0.1144 | 0.1440 | 0.1600 | 0.1395 | 0.3933 | 원 논문식 clean-only PI, 오탐 매우 큼 |
| Zhang attack-tuned | Deep learning forecaster + tuned threshold | 0.1111 | 0.1457 | 0.1661 | 0.1410 | 0.3404 | 공격 validation으로 threshold 맞춰도 오탐 큼 |
| Tufail/compare2 paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | 0.3126 | 0.4314 | 0.4906 | 0.4115 | 0.0479 | supervised baseline 중 가장 적절하지만 proposed보다 낮음 |

이 표를 본문 대표 결과표로 쓰면 된다.

## 4. Accuracy까지 포함한 Main Table

| Model | 방법론 타입 | Attack | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---:|---:|---:|---:|---:|
| Proposed | Two-stage residual ML detector | SA5 | 0.9736 | 0.8737 | 0.6107 | 0.7189 | 0.0052 |
| Proposed | Two-stage residual ML detector | SA8 | 0.9857 | 0.9024 | 0.8315 | 0.8655 | 0.0053 |
| Proposed | Two-stage residual ML detector | SA10 | 0.9905 | 0.9157 | 0.9125 | 0.9141 | 0.0049 |
| CNN-LSTM PI single-site aligned | Deep learning PI, clean-only | SA5 | 0.5789 | 0.0649 | 0.4893 | 0.1146 | 0.4159 |
| CNN-LSTM PI single-site aligned | Deep learning PI, clean-only | SA8 | 0.5807 | 0.0723 | 0.5519 | 0.1279 | 0.4176 |
| CNN-LSTM PI single-site aligned | Deep learning PI, clean-only | SA10 | 0.5820 | 0.0766 | 0.5883 | 0.1355 | 0.4184 |
| Zhang test7-aligned | Deep learning PI, clean-only | SA5 | 0.5995 | 0.0652 | 0.4643 | 0.1144 | 0.3925 |
| Zhang test7-aligned | Deep learning PI, clean-only | SA8 | 0.6059 | 0.0819 | 0.5951 | 0.1440 | 0.3935 |
| Zhang test7-aligned | Deep learning PI, clean-only | SA10 | 0.6094 | 0.0909 | 0.6680 | 0.1600 | 0.3941 |
| Zhang attack-tuned | Deep learning forecaster + tuned threshold | SA5 | 0.6458 | 0.0646 | 0.3973 | 0.1111 | 0.3396 |
| Zhang attack-tuned | Deep learning forecaster + tuned threshold | SA8 | 0.6524 | 0.0844 | 0.5321 | 0.1457 | 0.3405 |
| Zhang attack-tuned | Deep learning forecaster + tuned threshold | SA10 | 0.6564 | 0.0960 | 0.6143 | 0.1661 | 0.3411 |
| Tufail/compare2 paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | SA5 | 0.9181 | 0.2918 | 0.3365 | 0.3126 | 0.0478 |
| Tufail/compare2 paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | SA8 | 0.9271 | 0.3794 | 0.4997 | 0.4314 | 0.0479 |
| Tufail/compare2 paper_with_time | Supervised RF/MLP/CNN-LSTM ensemble | SA10 | 0.9321 | 0.4194 | 0.5908 | 0.4906 | 0.0479 |

본문에는 위 표가 조금 길 수 있으므로, 평균 F1/FPR 표를 먼저 넣고 이 표는 상세 결과표로 넣는 것을 추천한다.

## 5. Ablation에 넣을 모델

Ablation 표에는 아래 모델들을 넣는다.

| Model | 방법론 타입 | 넣는 이유 |
|---|---|---|
| Feature-matched RF+MLP+CNN-LSTM | Supervised ensemble on proposed residual features | 입력 feature까지 제안모델과 동일하게 맞춘 공정성 보강 실험 |
| Weather-only | Supervised classifier on weather/time features | 기상 feature만으로 직접 탐지가 어려움을 보여줌 |
| Weather + power_ratio | Supervised classifier on weather/time + measured output | 출력값을 직접 넣어도 일반화에 한계가 있음을 보여줌 |
| Tufail engineered | Supervised ensemble with engineered direct sensor features | feature engineering을 많이 해도 residual 구조를 못 넘는다는 방어 실험 |
| Proposed | Two-stage residual ML detector | 기준 성능 |

## 6. Feature-Matched 결과표

| Model | 방법론 타입 | SA5 F1 | SA8 F1 | SA10 F1 | Avg F1 | Avg FPR | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Proposed zone-wise LightGBM | Two-stage residual ML detector | 0.7189 | 0.8655 | 0.9141 | 0.8328 | 0.0051 | 제안모델 |
| Feature-matched RF+MLP+CNN-LSTM | Supervised ensemble on proposed residual features | 0.4729 | 0.7285 | 0.7974 | 0.6662 | 0.0027 | 같은 feature를 줘도 F1은 proposed보다 낮음 |

이 표가 “입력 feature까지 같게 해야 한다”는 지적에 대한 가장 직접적인 답이다.

## 7. Ablation 결과표

| Model | 방법론 타입 | SA5 F1 | SA8 F1 | SA10 F1 | Avg F1 | Avg FPR | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Weather-only | Supervised weather-only classifier | 0.1087 | 0.1087 | 0.1087 | 0.1087 | 0.1063 | 기상만 직접 분류하면 공격 신호가 약함 |
| Weather + power_ratio | Supervised weather+output classifier | 0.2342 | 0.3164 | 0.3668 | 0.3058 | 0.1016 | 출력값을 넣어도 external 일반화 한계 |
| Tufail engineered | Supervised engineered-feature ensemble | 0.2808 | 0.4042 | 0.4597 | 0.3816 | 0.0656 | feature를 늘려도 proposed보다 낮음 |
| Proposed | Two-stage residual ML detector | 0.7189 | 0.8655 | 0.9141 | 0.8328 | 0.0051 | residual 구조 효과 |

이 표는 “기상데이터만으로 직접 탐지하면 안 좋다”를 실패가 아니라 근거로 바꿔준다.

## 8. 본문에 쓰지 않는 것이 좋은 것

아래 결과는 본문 main comparison에 넣기보다는 appendix로 빼는 것을 추천한다.

| Model | 이유 |
|---|---|
| Zhang original reproduction in `ai` | test7 split과 평가 데이터가 다르다 |
| Original spatial CNN-LSTM reproduction | 2x2 spatial matrix 재현 성격이며 단일 6.0kW external 평가와 직접 정합되지 않는다 |
| Smoke test 결과 | sample/window cap이 있어서 정식 비교 수치가 아니다 |
| Weather-only only table 단독 강조 | 성능이 너무 낮으므로 단독 비교모델처럼 보이면 약해 보인다 |

Zhang original reproduction은 “논문 재현은 했다”는 근거로 appendix에 넣으면 된다. 본문 핵심 비교는 `test7-aligned` 결과를 사용한다.

## 9. 최종 해석 문장

본문용:

> 제안모델은 Zhang 및 Tufail 기반 비교모델보다 모든 SA 공격 강도에서 높은 F1과 낮은 FPR을 달성하였다. 특히 Zhang baseline은 공격 recall은 확보할 수 있으나 FPR과 day-level false alarm이 매우 높았고, Tufail-style supervised ensemble 역시 동일한 공격 학습 조건에서도 제안모델보다 낮은 F1과 높은 FPR을 보였다.

Ablation용:

> Weather-only classifier의 낮은 성능은 기상 feature만으로 SA 공격을 직접 분류하기 어렵다는 점을 보여준다. Weather+power_ratio 및 engineered-feature baseline에서도 성능 향상은 제한적이었으며, 이는 단순 센서 feature 직접 분류보다 기상 조건 기반 정상 발전량 예측과 residual dynamics를 결합하는 제안 구조가 효과적임을 뒷받침한다.

Feature-matched용:

> 입력 feature 차이에 따른 영향을 통제하기 위해, 제안모델과 동일한 residual/window/GHI feature를 RF, MLP, CNN-LSTM 및 soft-voting ensemble에 동일하게 제공하는 feature-matched 비교실험을 수행하였다. 동일 feature를 사용한 baseline은 기존 Tufail feature baseline보다 성능이 향상되었지만, 모든 SA 공격 강도에서 제안모델보다 낮은 F1을 보였다.

짧은 결론:

> 따라서 비교 결과는 제안모델의 성능 향상이 단순 feature 수 증가 때문이 아니라, 정상 발전량 예측 residual을 이용한 탐지 구조에서 나온 것임을 보여준다.
