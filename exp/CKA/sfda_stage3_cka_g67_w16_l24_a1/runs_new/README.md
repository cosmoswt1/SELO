# runs_new 실험 비교 (A1)

- 대상 경로: `exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new`
- 집계 기준: `eval_by_epoch/epoch_*/eval_metrics.json`
- 집계 run 수: 총 11개, eval 존재 8개

## 1) 종합 순위 (best adapted mIoU)

| rank | run | best_epoch | best_adapt_mIoU | best_delta(abs) | last_epoch | last_adapt_mIoU | last_delta(abs) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `v0-noupd-continue-noconst` | 10 | 54.1222 | 0.1017 | 10 | 54.1222 | 0.1017 |
| 2 | `v0-noupd` | 5 | 54.0790 | 0.0585 | 5 | 54.0790 | 0.0585 |
| 3 | `v0-du05do05--ep1du1` | 3 | 54.0550 | 0.0345 | 3 | 54.0550 | 0.0345 |
| 4 | `v1` | 3 | 54.0421 | 0.0216 | 4 | 54.0420 | 0.0215 |
| 5 | `v0-noupd-nodual-do002-lo5` | 4 | 54.0344 | 0.0139 | 4 | 54.0344 | 0.0139 |
| 6 | `v0-du05do05` | 2 | 54.0331 | 0.0127 | 2 | 54.0331 | 0.0127 |
| 7 | `a1-live-forcegate1-nosel-fresh-20260223_164943` | 2 | 54.0192 | -0.0013 | 2 | 54.0192 | -0.0013 |
| 8 | `v0` | 1 | 54.0185 | -0.0020 | 1 | 54.0185 | -0.0020 |

## 2) 에폭별 adapted mIoU

| run | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `a1-live-forcegate1-lowconf-shift-20260223_212123` | - | - | - | - | - | - | - | - | - | - |
| `a1-live-forcegate1-nosel-fresh-20260223_164943` | 54.0161 | 54.0192 | - | - | - | - | - | - | - | - |
| `v0` | 54.0185 | - | - | - | - | - | - | - | - | - |
| `v0-du05` | - | - | - | - | - | - | - | - | - | - |
| `v0-du05do01` | - | - | - | - | - | - | - | - | - | - |
| `v0-du05do05` | 54.0301 | 54.0331 | - | - | - | - | - | - | - | - |
| `v0-du05do05--ep1du1` | - | 54.0412 | 54.0550 | - | - | - | - | - | - | - |
| `v0-noupd` | 54.0295 | 54.0395 | 54.0557 | 54.0714 | 54.0790 | - | - | - | - | - |
| `v0-noupd-continue-noconst` | - | - | - | - | - | 54.0912 | 54.0991 | 54.1076 | 54.1157 | 54.1222 |
| `v0-noupd-nodual-do002-lo5` | 54.0288 | 54.0249 | 54.0320 | 54.0344 | - | - | - | - | - | - |
| `v1` | 54.0372 | 54.0377 | 54.0421 | 54.0420 | - | - | - | - | - | - |

## 3) 에폭별 delta(abs)=adapt-base

| run | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `a1-live-forcegate1-lowconf-shift-20260223_212123` | - | - | - | - | - | - | - | - | - | - |
| `a1-live-forcegate1-nosel-fresh-20260223_164943` | -0.0044 | -0.0013 | - | - | - | - | - | - | - | - |
| `v0` | -0.0020 | - | - | - | - | - | - | - | - | - |
| `v0-du05` | - | - | - | - | - | - | - | - | - | - |
| `v0-du05do01` | - | - | - | - | - | - | - | - | - | - |
| `v0-du05do05` | +0.0096 | +0.0127 | - | - | - | - | - | - | - | - |
| `v0-du05do05--ep1du1` | - | +0.0207 | +0.0345 | - | - | - | - | - | - | - |
| `v0-noupd` | +0.0090 | +0.0190 | +0.0352 | +0.0509 | +0.0585 | - | - | - | - | - |
| `v0-noupd-continue-noconst` | - | - | - | - | - | +0.0707 | +0.0786 | +0.0871 | +0.0952 | +0.1017 |
| `v0-noupd-nodual-do002-lo5` | +0.0083 | +0.0044 | +0.0115 | +0.0139 | - | - | - | - | - | - |
| `v1` | +0.0167 | +0.0172 | +0.0216 | +0.0215 | - | - | - | - | - | - |

## 4) Best epoch 기준 조건별 mIoU (adapt/base)

| run | best_epoch | fog | night | rain | snow |
|---|---:|---|---|---|---|
| `v0-noupd-continue-noconst` | 10 | 70.656/70.595 (+0.062) | 30.605/30.562 (+0.043) | 60.472/60.359 (+0.113) | 54.250/54.113 (+0.137) |
| `v0-noupd` | 5 | 70.618/70.595 (+0.024) | 30.613/30.562 (+0.051) | 60.422/60.359 (+0.063) | 54.179/54.113 (+0.065) |
| `v0-du05do05--ep1du1` | 3 | 70.602/70.595 (+0.007) | 30.617/30.562 (+0.054) | 60.389/60.359 (+0.030) | 54.143/54.113 (+0.030) |
| `v1` | 3 | 70.611/70.595 (+0.016) | 30.601/30.562 (+0.039) | 60.378/60.359 (+0.019) | 54.122/54.113 (+0.009) |
| `v0-noupd-nodual-do002-lo5` | 4 | 70.592/70.595 (-0.002) | 30.586/30.562 (+0.024) | 60.383/60.359 (+0.023) | 54.127/54.113 (+0.014) |
| `v0-du05do05` | 2 | 70.600/70.595 (+0.005) | 30.599/30.562 (+0.037) | 60.374/60.359 (+0.015) | 54.114/54.113 (+0.001) |
| `a1-live-forcegate1-nosel-fresh-20260223_164943` | 2 | 70.602/70.595 (+0.008) | 30.569/30.562 (+0.007) | 60.364/60.359 (+0.004) | 54.100/54.113 (-0.013) |
| `v0` | 1 | 70.594/70.595 (-0.000) | 30.583/30.562 (+0.021) | 60.361/60.359 (+0.002) | 54.095/54.113 (-0.018) |

## 5) Run별 핵심 파라미터

| run | delta_out | delta_upd | use_upd_loss | lambda_out_init | lambda_upd_init | dual_lr_out | dual_lr_upd | lambda_max | anchor_conf_gamma | anchor_conf_thresh | anchor_temperature | force_gate_one | gate_detach_align | lambda_select | select_score_norm | local_window_size | local_windows_total | local_windows_per_step | boundary_ratio_local | lr | weight_decay | batch_size | grad_accum_steps | workers | seed | eval_resize | eval_split | resume_ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a1-live-forcegate1-lowconf-shift-20260223_212123` | 0.03 | 0.1 | - | 0.5 | 0.2 | 0.03 | 0.01 | 10.0 | 3.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `a1-live-forcegate1-nosel-fresh-20260223_164943` | 0.02 | 0.01 | - | 1.0 | 1.0 | 0.05 | 0.05 | 10.0 | 1.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0` | 0.03 | 0.2 | - | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0-du05` | 0.03 | 0.5 | - | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0-du05do01` | 0.1 | 0.5 | - | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0-du05do05` | 0.5 | 0.5 | - | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0-du05do05--ep1du1` | 0.5 | 1.0 | - | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05do05/adapter_epoch_001.pth |
| `v0-noupd` | 0.5 | 0.5 | 0 | 0.2 | 0.1 | 0.05 | 0.05 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v0-noupd-continue-noconst` | 0.02 | 0.5 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 2.0 | 0.93 | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | /home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd/adapter_epoch_005_noconst.pth |
| `v0-noupd-nodual-do002-lo5` | 0.02 | 0.5 | 0 | 5.0 | 0.0 | 0.0 | 0.0 | 10.0 | 2.0 | - | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |
| `v1` | 0.02 | 0.5 | 0 | 5.0 | 0.0 | 0.0 | 0.0 | 10.0 | 2.0 | 0.93 | 1.0 | 1 | 1 | 0.0 | 1 | 16 | 10 | 10 | 0.6 | 0.0001 | 0.01 | 1 | 4 | 2 | 1 | 1080 | val | (empty) |

- 참고: 값이 `-`인 항목은 해당 run 시점 코드의 Args 로그에 그 키가 없던 경우입니다.

## 6) 비교 분석

- 최고 성능은 `v0-noupd-continue-noconst`이며 best adapted mIoU=54.1222 (epoch 10)입니다.
- `v0-noupd-continue-noconst`는 E6→E10에서 54.0912→54.1222로 단조 상승했고, delta(abs)는 +0.0707→+0.1017로 확대되었습니다.
- `v0-noupd` 대비 best adapted mIoU가 +0.0432p(54.0790→54.1222) 추가 개선되었습니다.
- `v0-noupd`는 E1→E5에서 54.0295→54.0790로 단조 상승하며, delta(abs)도 +0.0090→+0.0585로 확대되었습니다.
- `v0-noupd-nodual-do002-lo5`(dual off, out 강제)는 개선폭은 작지만 모든 평가 epoch에서 양의 delta를 유지했습니다(최대 +0.0139).
- `v1`(anchor_conf_thresh=0.93)은 E1부터 +0.0167의 양의 시작점을 가지며 E3에서 +0.0216으로 피크 후 E4에서 plateau(+0.0215) 양상입니다.
- 초기 `a1-live-forcegate1-nosel-fresh-20260223_164943`는 E1/E2 모두 음수 delta(-0.0044, -0.0013)로, 이후 설정 대비 열세입니다.
- 조건별로는 공통적으로 `night` 개선폭이 가장 작고, `rain/snow`에서 상대적으로 이득이 발생하는 경향이 보입니다.
- 상위 run은 +0.10x 수준까지 개선됐지만, 하위 run은 여전히 +0.00x 또는 음수 구간이 남아 있어 동일 seed 다회 반복으로 분산(표준편차) 검증이 필요합니다.

## 7) Eval 누락 run

- `a1-live-forcegate1-lowconf-shift-20260223_212123`: `eval_by_epoch/*/eval_metrics.json` 없음 (중단/미실행 가능).
- `v0-du05`: `eval_by_epoch/*/eval_metrics.json` 없음 (중단/미실행 가능).
- `v0-du05do01`: `eval_by_epoch/*/eval_metrics.json` 없음 (중단/미실행 가능).

## 8) Run별 Args 원문 (마지막 기록)

### `a1-live-forcegate1-lowconf-shift-20260223_212123`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/a1-live-forcegate1-lowconf-shift-20260223_212123 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.03 delta_upd=0.1 lambda_out_init=0.5 lambda_upd_init=0.2 dual_lr_out=0.03 dual_lr_upd=0.01 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=3.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `a1-live-forcegate1-nosel-fresh-20260223_164943`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/a1-live-forcegate1-nosel-fresh-20260223_164943 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.02 delta_upd=0.01 lambda_out_init=1.0 lambda_upd_init=1.0 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=1.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.03 delta_upd=0.2 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-du05`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.03 delta_upd=0.5 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-du05do01`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05do01 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.1 delta_upd=0.5 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-du05do05`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05do05 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.5 delta_upd=0.5 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-du05do05--ep1du1`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05do05--ep1du1 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.5 delta_upd=1.0 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt=exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-du05do05/adapter_epoch_001.pth eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-noupd`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.5 delta_upd=0.5 use_upd_loss=0 lambda_out_init=0.2 lambda_upd_init=0.1 dual_lr_out=0.05 dual_lr_upd=0.05 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-noupd-continue-noconst`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.02 delta_upd=0.5 use_upd_loss=0 lambda_out_init=0.0 lambda_upd_init=0.0 dual_lr_out=0.0 dual_lr_upd=0.0 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_conf_thresh=0.93 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=10 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd/adapter_epoch_005_noconst.pth eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v0-noupd-nodual-do002-lo5`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-nodual-do002-lo5 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.02 delta_upd=0.5 use_upd_loss=0 lambda_out_init=5.0 lambda_upd_init=0.0 dual_lr_out=0.0 dual_lr_upd=0.0 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```

### `v1`
```text
acdc_root=/mnt/d/ACDC output_dir=/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v1 conditions=['fog', 'night', 'rain', 'snow'] resize=1072 crop_size=1072 segformer_model=nvidia/segformer-b5-finetuned-cityscapes-1024-1024 dino_model=facebook/dinov3-vitl16-pretrain-lvd1689m dino_layer=24 num_classes=19 adapter_bottleneck=128 gate_bias_init=-4.0 delta_out=0.02 delta_upd=0.5 use_upd_loss=0 lambda_out_init=5.0 lambda_upd_init=0.0 dual_lr_out=0.0 dual_lr_upd=0.0 lambda_max=10.0 gate_detach_align=1 lambda_select=0.0 select_score_norm=1 anchor_conf_gamma=2.0 anchor_conf_thresh=0.93 anchor_temperature=1.0 force_gate_one=1 local_window_size=16 local_windows_total=10 local_windows_per_step=10 boundary_ratio_local=0.6 epochs=5 batch_size=1 workers=2 grad_accum_steps=4 lr=0.0001 weight_decay=0.01 max_grad_norm=1.0 amp=True seed=1 log_every=20 max_steps=0 overfit_one_batch=0 overfit_fixed_sampling=0 strict_dino_resolution=1 diag_heavy_interval=50 diag_den_warn=0.0001 diag_den_critical=1e-06 resume_ckpt= eval_every_epoch=1 eval_split=val eval_test_gt_dir= eval_resize=1080 eval_batch_size=1 eval_workers=0
```
