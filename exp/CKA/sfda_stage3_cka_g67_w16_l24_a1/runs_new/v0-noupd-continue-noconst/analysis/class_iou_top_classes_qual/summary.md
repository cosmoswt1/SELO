# Top Class Qualitative Panels

## Preprocessing / Label Mapping / Resize-Crop Rule

- Split: `val`
- Conditions: `rain, snow`
- Resize rule: shorter side -> `1080`, then eval path에서 32배수 pad
- Crop rule: 없음 (evaluation에서는 random crop 미사용)
- Label mapping: Cityscapes trainIds (`0..18`), ignore=`255`
- Panel crop rule: target GT/pred union bbox를 기준으로 `1.8x` 확장, 최소 crop 크기 적용
- Panel legend: GT target=blue, TP=green, FP=red, FN=blue, adapt-added=yellow, adapt-removed=magenta

## mIoU (%)

| group | num_images | baseline | adapted | delta(abs) |
|---|---:|---:|---:|---:|
| overall | 406 | 54.02 | 54.12 | 0.10 |
| rain | 100 | 60.36 | 60.47 | 0.11 |
| snow | 100 | 54.11 | 54.25 | 0.14 |

## Selected Panels

| condition | class | selection | direction | class delta(cond) | image delta | dominant effect | panel |
|---|---|---|---|---:|---:|---|---|
| rain | traffic sign | best | negative | -0.26 | -1.57 | TP 개선보다 FP 증가가 커 false positive expansion이 우세함. | [rain_traffic_sign_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_traffic_sign_negative.png) |
| rain | traffic sign | fp_alert | negative | -0.26 | 0.00 | GT가 없는 장면에서 false positive가 늘어남. | [rain_traffic_sign_negative_fp_alert.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_traffic_sign_negative_fp_alert.png) |
| rain | truck | best | negative | -0.28 | -1.76 | TP 개선보다 FP 증가가 커 false positive expansion이 우세함. | [rain_truck_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_truck_negative.png) |
| rain | truck | fp_alert | negative | -0.28 | 0.00 | GT가 없는 장면에서 false positive가 늘어남. | [rain_truck_negative_fp_alert.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_truck_negative_fp_alert.png) |
| rain | bus | best | positive | 0.32 | 1.92 | 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. | [rain_bus_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_bus_positive.png) |
| rain | person | best | positive | 0.19 | 0.85 | FN 감소가 커서 경계/누락 복원이 주효함. | [rain_person_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_person_positive.png) |
| rain | pole | best | positive | 0.36 | 2.02 | 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. | [rain_pole_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_pole_positive.png) |
| rain | train | best | positive | 1.06 | 1.18 | 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. | [rain_train_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_train_positive.png) |
| snow | bicycle | best | negative | -0.15 | -2.98 | FN 증가가 커서 object extent가 줄거나 경계를 놓침. | [snow_bicycle_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_bicycle_negative.png) |
| snow | bicycle | fp_alert | negative | -0.15 | 0.00 | GT가 없는 장면에서 false positive가 늘어남. | [snow_bicycle_negative_fp_alert.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_bicycle_negative_fp_alert.png) |
| snow | bus | best | positive | 0.32 | 1.65 | FN 감소가 커서 경계/누락 복원이 주효함. | [snow_bus_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_bus_positive.png) |
| snow | person | best | positive | 0.17 | 1.45 | FN 감소가 커서 경계/누락 복원이 주효함. | [snow_person_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_person_positive.png) |
| snow | pole | best | positive | 0.23 | 5.45 | 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. | [snow_pole_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_pole_positive.png) |
| snow | truck | best | positive | 1.29 | 2.30 | 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. | [snow_truck_positive.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_truck_positive.png) |

## Interpretation

- `rain / traffic sign / best`: class IoU delta=-0.26, selected image delta=-1.57, TP `6920->6927`, FP `11428->12336`, FN `1011->1004`. 판정: TP 개선보다 FP 증가가 커 false positive expansion이 우세함. panel=`panels/rain_traffic_sign_negative.png`
- `rain / traffic sign / fp_alert`: class IoU delta=-0.26, selected image delta=0.00, TP `0->0`, FP `930->978`, FN `0->0`. 판정: GT가 없는 장면에서 false positive가 늘어남. panel=`panels/rain_traffic_sign_negative_fp_alert.png`
- `rain / truck / best`: class IoU delta=-0.28, selected image delta=-1.76, TP `14603->14579`, FP `3027->3582`, FN `4251->4275`. 판정: TP 개선보다 FP 증가가 커 false positive expansion이 우세함. panel=`panels/rain_truck_negative.png`
- `rain / truck / fp_alert`: class IoU delta=-0.28, selected image delta=0.00, TP `0->0`, FP `8761->9716`, FN `0->0`. 판정: GT가 없는 장면에서 false positive가 늘어남. panel=`panels/rain_truck_negative_fp_alert.png`
- `rain / bus / best`: class IoU delta=0.32, selected image delta=1.92, TP `2806->2896`, FP `7->8`, FN `1853->1763`. 판정: 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. panel=`panels/rain_bus_positive.png`
- `rain / person / best`: class IoU delta=0.19, selected image delta=0.85, TP `99->103`, FP `445->384`, FN `604->600`. 판정: FN 감소가 커서 경계/누락 복원이 주효함. panel=`panels/rain_person_positive.png`
- `rain / pole / best`: class IoU delta=0.36, selected image delta=2.02, TP `32897->34348`, FP `3830->3948`, FN `32323->30872`. 판정: 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. panel=`panels/rain_pole_positive.png`
- `rain / train / best`: class IoU delta=1.06, selected image delta=1.18, TP `226632->229913`, FP `7172->7337`, FN `33197->29916`. 판정: 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. panel=`panels/rain_train_positive.png`
- `snow / bicycle / best`: class IoU delta=-0.15, selected image delta=-2.98, TP `97->88`, FP `1->1`, FN `204->213`. 판정: FN 증가가 커서 object extent가 줄거나 경계를 놓침. panel=`panels/snow_bicycle_negative.png`
- `snow / bicycle / fp_alert`: class IoU delta=-0.15, selected image delta=0.00, TP `0->0`, FP `23599->23878`, FN `0->0`. 판정: GT가 없는 장면에서 false positive가 늘어남. panel=`panels/snow_bicycle_negative_fp_alert.png`
- `snow / bus / best`: class IoU delta=0.32, selected image delta=1.65, TP `1476->1549`, FP `0->0`, FN `2955->2882`. 판정: FN 감소가 커서 경계/누락 복원이 주효함. panel=`panels/snow_bus_positive.png`
- `snow / person / best`: class IoU delta=0.17, selected image delta=1.45, TP `9->10`, FP `3->2`, FN `66->65`. 판정: FN 감소가 커서 경계/누락 복원이 주효함. panel=`panels/snow_person_positive.png`
- `snow / pole / best`: class IoU delta=0.23, selected image delta=5.45, TP `128->157`, FP `9->12`, FN `378->349`. 판정: 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. panel=`panels/snow_pole_positive.png`
- `snow / truck / best`: class IoU delta=1.29, selected image delta=2.30, TP `366808->379488`, FP `1380->1632`, FN `175158->162478`. 판정: 예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함. panel=`panels/snow_truck_positive.png`

## Qualitative 실패 케이스 3장
- Case 1: `rain / traffic sign` | image delta=-1.57, TP `6920->6927`, FP `11428->12336`, FN `1011->1004` | 설명: TP 개선보다 FP 증가가 커 false positive expansion이 우세함. | file=[rain_traffic_sign_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_traffic_sign_negative.png)
- Case 2: `rain / truck` | image delta=-1.76, TP `14603->14579`, FP `3027->3582`, FN `4251->4275` | 설명: TP 개선보다 FP 증가가 커 false positive expansion이 우세함. | file=[rain_truck_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/rain_truck_negative.png)
- Case 3: `snow / bicycle` | image delta=-2.98, TP `97->88`, FP `1->1`, FN `204->213` | 설명: FN 증가가 커서 object extent가 줄거나 경계를 놓침. | file=[snow_bicycle_negative.png](/home/kevinlee01/SELO/exp/CKA/sfda_stage3_cka_g67_w16_l24_a1/runs_new/v0-noupd-continue-noconst/analysis/class_iou_top_classes_qual/panels/snow_bicycle_negative.png)

## 다음 액션 (1)
- 같은 장면들에 대해 gate map 또는 confidence map을 겹쳐서, 회복/붕괴가 실제로 어느 영역에서 시작되는지 1회 확인.
