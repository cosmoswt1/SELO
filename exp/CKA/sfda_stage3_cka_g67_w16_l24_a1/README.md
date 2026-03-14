# SFDA Stage3 CKA (Local-Only, L24 post-LN)

## 목적
- SegFormer B5 stage3 feature `F16`를 adapter로 보정해 DINOv3 L24 토큰 구조와 정렬한다.
- 학습 손실은 **Local CKA 하나만** 사용한다.
- DINO teacher는 학습 시에만 사용하며, 평가/추론에서는 사용하지 않는다.

## 핵심 규칙
- Local CKA 윈도우는 매 step `10개` 사용 (`local_windows_total=10`, `local_windows_per_step=10` 고정).
- Local CKA 전처리는 **토큰축 centering만** 적용한다.
  - 채널 평균 제거 없음
  - L2 정규화 없음
- DINO L24 feature는 **post-LN (`last_hidden_state`)**를 사용한다.

## 손실
- `total_loss = loss_local`
- `loss_local = mean(1 - cka_local_each)`
- CKA 흐름: `token-centering -> Gram(NxN) -> normalized HSIC`

## 실행
### 1) 학습
```bash
bash exp/CKA/sfda_stage3_cka_g67_w16_l24/run_train.sh
```

### 2) 평가 (기본 val)
```bash
bash exp/CKA/sfda_stage3_cka_g67_w16_l24/run_eval.sh
```

### 3) test 평가 (외부 GT 사용)
```bash
TEST_GT_DIR=/path/to/test_gt bash exp/CKA/sfda_stage3_cka_g67_w16_l24/run_eval.sh --split test
```

## 산출물
- 학습 기본
  - `runs/<run_id>/train.log`
  - `runs/<run_id>/adapter.pth`
- 진단 CSV
  - `runs/<run_id>/diagnostics/den_stats.csv`
  - `runs/<run_id>/diagnostics/local_cka_stats.csv`
- 상시 추세 플롯
  - `runs/<run_id>/diagnostics/plots/den_min_trend.png`
  - `runs/<run_id>/diagnostics/plots/den_small_count_trend.png`
  - `runs/<run_id>/diagnostics/plots/local_cka_trend.png`
- 중량 진단(주기/이상치 트리거)
  - `runs/<run_id>/diagnostics/heavy/step_*/grad_conflict.json`
  - `runs/<run_id>/diagnostics/heavy/step_*/window_metrics.csv`
  - `runs/<run_id>/diagnostics/heavy/step_*/cosine_matrix.csv`
  - `runs/<run_id>/diagnostics/heavy/step_*/cosine_matrix.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/grad_norms.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/den_hist.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/den_sorted.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/cka_sorted.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/den_map.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/cka_map.png`
  - `runs/<run_id>/diagnostics/heavy/step_*/input_windows.png`

## 규칙
- GPU-only: 실행 전 `nvidia-smi` 확인, CUDA 미사용 시 중단.
- 기본 평가는 `val split`.
- test split에서 GT가 없으면 `No GT`로 정상 처리.
