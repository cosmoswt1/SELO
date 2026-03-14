# Class-wise IoU By Weather

## Preprocessing / Label Mapping / Resize-Crop Rule

- Split: `val`
- Resize rule: shorter side -> `1080`, then eval path에서 32배수 pad
- Crop rule: 없음 (evaluation에서는 random crop 미사용)
- Label mapping: Cityscapes trainIds (`0..18`), ignore=`255`
- Comparison: baseline(segformer base) vs adapted(Stage3 CKA adapter)
- Group split: `large_structure` = road/sidewalk/building/wall/fence/vegetation/terrain/sky, `small_object` = pole/traffic light/traffic sign/person/rider/car/truck/bus/train/motorcycle/bicycle

## Weather Overview

| condition | class_group | num_valid_classes | mean_delta_abs | median_delta_abs | max_delta_abs | min_delta_abs |
|---|---|---:|---:|---:|---:|---:|
| fog | large_structure | 8 | -0.010 | 0.002 | 0.094 | -0.108 |
| fog | small_object | 11 | 0.114 | -0.029 | 0.658 | -0.146 |
| night | large_structure | 8 | 0.045 | -0.040 | 0.355 | -0.138 |
| night | small_object | 11 | 0.038 | 0.033 | 0.168 | -0.073 |
| rain | large_structure | 8 | 0.062 | -0.003 | 0.444 | -0.045 |
| rain | small_object | 11 | 0.150 | 0.126 | 1.064 | -0.278 |
| snow | large_structure | 8 | 0.048 | 0.023 | 0.227 | -0.096 |
| snow | small_object | 11 | 0.202 | 0.069 | 1.287 | -0.146 |

## Rain / Snow Focus

### rain

| group | mean_delta_abs | median_delta_abs | max_delta_abs | min_delta_abs |
|---|---:|---:|---:|---:|
| large_structure | 0.062 | -0.003 | 0.444 | -0.045 |
| small_object | 0.150 | 0.126 | 1.064 | -0.278 |

- large_structure positive: sidewalk(0.444), road(0.053), fence(0.049), vegetation(0.013), sky(-0.003)
- large_structure negative: wall(-0.045), building(-0.007), terrain(-0.005), sky(-0.003), vegetation(0.013)
- small_object positive: train(1.064), pole(0.356), bus(0.323), person(0.193), motorcycle(0.133)
- small_object negative: truck(-0.278), traffic sign(-0.255), rider(-0.040), traffic light(-0.024), car(0.047)
- overall top positive classes: train(1.064), sidewalk(0.444), pole(0.356), bus(0.323), person(0.193)
- overall top negative classes: truck(-0.278), traffic sign(-0.255), wall(-0.045), rider(-0.040), traffic light(-0.024)
- 해석: large_structure 평균 delta=0.062, small_object 평균 delta=0.150 이므로 semantic recovery 쪽 신호가 더 강함.

### snow

| group | mean_delta_abs | median_delta_abs | max_delta_abs | min_delta_abs |
|---|---:|---:|---:|---:|
| large_structure | 0.048 | 0.023 | 0.227 | -0.096 |
| small_object | 0.202 | 0.069 | 1.287 | -0.146 |

- large_structure positive: sidewalk(0.227), wall(0.170), fence(0.027), building(0.026), road(0.023)
- large_structure negative: terrain(-0.096), vegetation(-0.008), sky(0.015), road(0.023), building(0.026)
- small_object positive: truck(1.287), bus(0.321), motorcycle(0.295), pole(0.233), person(0.165)
- small_object negative: bicycle(-0.146), traffic light(-0.073), rider(0.014), traffic sign(0.017), train(0.044)
- overall top positive classes: truck(1.287), bus(0.321), motorcycle(0.295), pole(0.233), sidewalk(0.227)
- overall top negative classes: bicycle(-0.146), terrain(-0.096), traffic light(-0.073), vegetation(-0.008), rider(0.014)
- 해석: large_structure 평균 delta=0.048, small_object 평균 delta=0.202 이므로 semantic recovery 쪽 신호가 더 강함.

## Next Action

- `rain/snow`에서 delta가 가장 큰 클래스만 골라, boundary 오류인지 texture 오류인지 qualitative panel 1회 추가 점검.
