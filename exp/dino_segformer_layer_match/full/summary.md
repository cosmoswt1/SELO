# DINO Layer ↔ SegFormer Stage Matching (ACDC all images, ref included)

## Preprocessing / Rules
- Dataset: `/mnt/d/ACDC`
- Conditions: `['fog', 'night', 'rain', 'snow']`
- Splits used: `train, val, test`
- Resize/Crop: shorter side `1072` -> center crop `1072x1072` (SegFormer/DINO 공통 입력)
- Token grid for CKA/SSM: adaptive pooled to `67x67`
- Token grid for visualization: `67x67` (native DINO grid if `--viz_grid_size 0`)
- Label mapping: N/A (feature similarity task, no semantic labels)
- Resize-crop rule: no random crop/flip, resize + pad only

## Token-Map Visualization Protocol
- 1) 한 점(앵커 토큰) 선택: `anchor_x/anchor_y` 비율로 grid 위치를 정하고 입력 이미지에 빨간 십자가로 표시
- 2) 코사인 유사도 계산: 같은 레이어의 모든 토큰 `z_q`와 앵커 토큰 `z_p`에 대해 `cos(z_p, z_q)`
- 3) 2D heatmap: 유사도 벡터를 `[H_p, W_p]`로 reshape 후 min-max normalize, `viridis` colormap 적용

## Data Count
- Total images: **4006**
- By condition: `fog=1000`, `night=1006`, `rain=1000`, `snow=1000`
- By split dir: `train=1600`, `val=406`, `test=2000`, `train_ref=0`, `val_ref=0`, `test_ref=0`
- By ref flag: `non_ref=4006`, `ref=0`

## Best Layer Mapping
| SegFormer Stage | Best DINO layer (CKA) | CKA | Best DINO layer (SSM) | SSM |
|---|---:|---:|---:|---:|
| stage1 | 6 | 0.780021 | 6 | 0.799037 |
| stage2 | 12 | 0.663252 | 12 | 0.781494 |
| stage3 | 24 | 0.635065 | 15 | 0.770631 |
| stage4 | 24 | 0.726037 | 15 | 0.711459 |

## Qualitative Low-Agreement Cases (Top 3)
- Case 1: score=0.545002, path=`/mnt/d/ACDC/rgb_anon_trainvaltest/rgb_anon/night/test/GOPR0364/GOPR0364_frame_000795_rgb_anon.png`
- Case 2: score=0.545442, path=`/mnt/d/ACDC/rgb_anon_trainvaltest/rgb_anon/rain/train/GP020400/GP020400_frame_000325_rgb_anon.png`
- Case 3: score=0.584130, path=`/mnt/d/ACDC/rgb_anon_trainvaltest/rgb_anon/rain/test/GOPR0572/GOPR0572_frame_000372_rgb_anon.png`

## Figures
- `figures/cka_overall_heatmap.png`
- `figures/ssm_overall_heatmap.png`
- `figures/best_layer_per_stage.png`
- `viz_stream/latest.png` (실시간 최신 토큰 시각화)
- `viz_stream/viz_*.png` (주기 저장 스냅샷)
- `qualitative/low_agreement_01.png` ~ `qualitative/low_agreement_03.png`

## Next Action (1)
- stage별로 선택된 best DINO layer를 고정해, stage feature alignment loss를 단계별 가중치로 넣은 1회 학습 실험을 수행.
