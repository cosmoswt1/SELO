# SFDA Stage3 CKA A1 아키텍처 (Trust-Region)

## 1. 기준 구현 파일
- `exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/model_cka_v1.py`
- `exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/loss_cka_v1.py`
- `exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/train_stage3_cka.py`
- `exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/eval_stage3_cka.py`
- `models/segformer_backbone.py`
- `models/dino_teacher.py`

## 2. 학습 대상과 고정 대상
- 학습 파라미터: `stage3_adapter`, `stage3_gate`
- 고정 파라미터: SegFormer backbone+decoder, DINO teacher
- 학습 중 모드 제어:
- `model.train()` 호출 후 backbone/decoder는 `eval()`로 고정해 dropout/확률층 노이즈를 막음
- `stage3_adapter`/`stage3_gate`만 `train()` 유지

## 3. Stage3 모듈 구성
- Adapter: `BottleneckDWResidualAdapter`
- 구조: `1x1 reduce -> 3x3 depthwise -> GELU -> 1x1 expand`
- 안정 초기화: 마지막 `expand` 가중치/바이어스 0으로 시작
- Gate: `SpatialGate`
- 구조: `depthwise conv -> GroupNorm -> 1x1 proj -> sigmoid`
- 출력: `g in [0,1]`, shape `[B,1,H,W]`
- 초기화: `bias_init=-4.0` 기본값으로 gate를 초기에 거의 닫힘 상태로 시작

## 4. Forward 모드 (핵심)
`Stage3CKAModel.forward(...)`는 `adapter_enabled`로 baseline/adapt 경로를 나눔.

### 4.1 Baseline 경로 (`adapter_enabled=False`)
- `f3_raw`만 사용하고 adapter/gate 갱신 없음
- `f3_pred = f3_raw`, `f3_align = f3_raw`
- 반환 `logits`는 output anchor의 기준 분포 `p0` 생성에 사용

### 4.2 Adapt 경로 (`adapter_enabled=True`)
- `delta = stage3_adapter(f3_raw)`
- `g = stage3_gate(f3_raw)` (단, `force_gate_one=1`이면 `g=1`)
- `update = g * delta` (alpha 항은 제거되어 항상 1)
- `f3_pred = f3_raw + update` (decoder logits 생성용)
- `f3_align = f3_raw + g_align * delta`
- `g_align = g.detach()` if `gate_detach_for_align=1`, else `g`
- 목적: `L_align`이 gate를 직접 키우는 gain knob가 되지 않게 차단

### 4.3 반환 dict 주요 키
- `logits`
- `stage3_raw`
- `delta`
- `gate`
- `update`
- `stage3_adapt` (`f3_pred`)
- `stage3_align` (`f3_align`)
- `dino_feat` (`use_dino=True`일 때)

## 5. DINO feature 추출 규칙
- `dino_layer < last`: `hidden_states[dino_layer]` (pre-LN)
- `dino_layer == last(기본 24)`: `last_hidden_state` (post-LN)
- 기본 실험(`dino_layer=24`)은 post-LN 사용

## 6. Loss 구성 (A1)
학습 총손실:

`total_loss = L_align + lambda_out * L_out + lambda_upd * L_upd + lambda_select * L_select`

### 6.1 `L_align` (Local CKA)
- 입력: `stage3_align` vs `dino_feat`
- window tensor shape: `[B,S,N,C]` (`S=10`, `N=window_size^2`)
- CKA 전처리:
- **토큰 축 centering만 사용**
- `Xc = X - mean(X, dim=tokens)`
- `Yc = Y - mean(Y, dim=tokens)`
- CKA 경로에서 token/channel L2 정규화는 사용하지 않음
- Gram:
- `Gx = Xc @ Xc^T`
- `Gy = Yc @ Yc^T`
- normalized HSIC:
- `hsic = <Gx,Gy>`
- `nx = ||Gx||_F`, `ny = ||Gy||_F`
- `den = max(nx*ny, eps)`
- `cka = hsic / den`
- loss:
- `L_align = mean(1 - cka_local_each)`

### 6.2 `L_out` (Do-No-Harm output anchor)
- baseline logits: `logits0 = model(..., adapter_enabled=False)` (no_grad)
- adapt logits: `logits1 = model(..., adapter_enabled=True)` (grad on)
- temperature `T = anchor_temperature`
- `p0 = softmax(logits0 / T)`
- confidence:
- `conf = max_c p0(c|u)`
- `conf = conf^gamma` (`gamma = anchor_conf_gamma`, default 1.0)
- KL map:
- `kl_map = sum_c p0 * (log p0 - log_softmax(logits1/T))`
- weighted metric:
- `out_metric = sum(conf * kl_map) / (sum(conf) + eps)`
- hinge-squared:
- `out_violation = relu(out_metric - delta_out)`
- `L_out = out_violation^2`

### 6.3 `L_upd` (minimal update energy)
- `upd_metric = mean(update^2) / (mean(stage3_raw^2).detach() + eps)`
- hinge-squared:
- `upd_violation = relu(upd_metric - delta_upd)`
- `L_upd = upd_violation^2`

### 6.4 `L_select` (gate selection objective)
- 목적: `gate_detach_align=1`을 유지하면서도 gate가 mismatch 큰 위치를 열도록 양의 유인 제공
- mismatch score map:
- `s = div_map(stage3_align, dino_feat)` (SSM divergence, detach)
- 옵션 `select_score_norm=1`이면 배치별 평균으로 정규화:
- `s_norm = s / (mean(s)+eps)`
- selection score:
- `select_score = mean(g * s_norm)` (`g`는 `stage3_gate` 출력)
- loss:
- `L_select = -select_score`
- `lambda_select`로 강도 제어 (`run_train.sh` 기본 0.5)

## 7. Dual update (`lambda_out`, `lambda_upd`)
- 업데이트 시점: optimizer step 경계(gradient accumulation 반영 후)
- micro-step 동안 detached metric 평균을 사용
- 규칙:
- `lambda_out <- clip(lambda_out + dual_lr_out * (out_metric_avg - delta_out), 0, lambda_max)`
- `lambda_upd <- clip(lambda_upd + dual_lr_upd * (upd_metric_avg - delta_upd), 0, lambda_max)`
- 체크포인트 저장/복원 포함:
- `lambda_out`, `lambda_upd`, `global_step`, `update_step`, `epoch`, `optimizer`, `scaler`

## 8. Local window 샘플링
- `local_windows_total=10`, `local_windows_per_step=10` 고정
- SSM divergence map 기반 score 생성 후 NMS로 window 선택
- NMS 제약: 기본 IoU 임계 `0.2`
- 참고: sampler 내부 score 계산에는 divergence 안정화를 위해 feature channel 정규화가 사용되지만, CKA loss 전처리는 centering-only 유지

## 9. 로그 필드 해석
`train.log`의 `MICRO`/`UPDATE`에 아래가 출력됨.

- `align`: `L_align`
- `out`: `L_out` (패널티 loss)
- `upd`: `L_upd` (패널티 loss)
- `select`: `L_select` (gate selection loss)
- `select_score`: `mean(g*s)` (높을수록 mismatch 위치를 gate가 잘 여는 상태)
- `div_mean`: selection에 쓰인 mismatch map 평균
- `out_metric`, `upd_metric`: 제약 지표 원값
- `out_violation`, `upd_violation`: 힌지 위반량
- `lambda_out`, `lambda_upd`: dual 변수
- `cka_local`: local CKA 평균
- `gate_mean`, `gate_max`: gate 모니터링

중요:
- `out/upd`가 0이어도 버그가 아닐 수 있음
- `out_metric <= delta_out`, `upd_metric <= delta_upd`이면 힌지 위반이 0이라 패널티 loss도 0으로 출력됨

## 10. mIoU 보고 경로
- 학습 중 step 로그는 주로 CKA/constraint 계열 지표를 출력
- mIoU는 epoch 종료 후 자동 eval(`eval_every_epoch`)에서 계산
- `EVAL-EPOCH e=... base=... adapt=... delta=...` 형태로 `train.log`에 기록
- 상세 결과:
- `eval_by_epoch/epoch_xxx/summary.md`
- `eval_by_epoch/epoch_xxx/results.csv`
- `eval_by_epoch/epoch_xxx/eval.log`
- `eval_by_epoch/epoch_xxx/eval_metrics.json`

## 11. 산출물
- 학습:
- `runs_new/<run_id>/train.log`
- `runs_new/<run_id>/adapter_epoch_XXX.pth`
- `runs_new/<run_id>/adapter.pth`
- 진단 CSV/플롯:
- `diagnostics/den_stats.csv`
- `diagnostics/local_cka_stats.csv`
- `diagnostics/window_sampling_stats.csv`
- `diagnostics/plots/den_min_trend.png`
- `diagnostics/plots/den_small_count_trend.png`
- `diagnostics/plots/local_cka_trend.png`
- `diagnostics/plots/select_score_trend.png`
- `diagnostics/plots/gate_mean_trend.png`
- `diagnostics/plots/gate_local_quantile_trend.png`
- `diagnostics/plots/update_rms_trend.png`
- `diagnostics/plots/suppression_ratio_trend.png`
- 중량 진단:
- `diagnostics/heavy/step_*/grad_conflict.json`
- `diagnostics/heavy/step_*/window_metrics.csv`
- `diagnostics/heavy/step_*/cosine_matrix.csv`
- `diagnostics/heavy/step_*/gate_heatmap.png`
- `diagnostics/heavy/step_*/gate_overlay.png`
- `diagnostics/heavy/step_*/gate_selected_map.png`
- `diagnostics/heavy/step_*/gate_align_heatmap.png`
- `diagnostics/heavy/step_*/gate_align_overlay.png`
- `diagnostics/heavy/step_*/gate_align_selected_map.png`
- `diagnostics/heavy/step_*/*.png`

## 12. 현재 레포 정책 요약
- 이 A1 레포는 trust-region 학습 경로를 기본으로 사용
- legacy 손실 경로(`loss_local + lambda_update*loss_update + lambda_gate*loss_gate`)는 포함하지 않음
- token L2 정규화는 CKA 경로에서 사용하지 않음
