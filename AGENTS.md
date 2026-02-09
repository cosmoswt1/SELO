# AGENTS.md — Project Rules (SELO / ACDC)

## 0) Language (Hard Rule)
- 모든 답변/설명/계획/요약은 **한국어**로 작성한다.
- 표/코드블록은 영어가 섞여도 되지만, 주변 설명은 한국어로 유지한다.

## 1) GPU-Only Policy (Hard Rule)
- 학습/평가/특징추출은 **항상 CUDA GPU로만 실행**한다. 가상환경은 이미 구축된 conda selo 이용
- CPU fallback(대신 CPU로 돌리기)은 **절대 금지**.
- CUDA GPU가 없거나 사용할 수 없으면:
  1) 어떤 체크에서 실패했는지
  2) 확인 커맨드 (nvidia-smi, python -c "import torch; print(torch.cuda.is_available())")
  3) 해결 방법
  을 한국어로 제시하고 작업을 중단한다.
- 실행 전 `nvidia-smi` 확인을 기본으로 한다.

## 2) Execution / Safety
- 어떤 명령을 실행하거나 파일을 대량 수정하기 전, 아래를 먼저 제시한다:
  - 실행할 커맨드(정확히)
  - 변경될 파일 목록
  - 결과 산출물 경로
- 실험/평가는 항상 결과 폴더를 만들고 그 안에 저장한다.
- “추측”으로 경로를 만들지 말고, 경로/파일 존재 여부를 먼저 확인한다.

## 3) ACDC Dataset / Split Convention
- 기본 평가는 **val split**을 사용한다.
- ACDC의 test split은 GT가 없을 수 있으므로,
  - test에서 "0 images / No GT"가 나오면 실패가 아니라 정상 상황으로 처리한다.
  - test 평가가 필요하면 GT 경로를 사용자에게 요청하거나, 옵션(`--test_gt_dir`)으로 받는다.

## 4) Output Artifacts (Always)
- baseline/eval 결과는 항상 아래 파일을 생성한다:
  - summary.md (표 포함)
  - results.csv
  - eval.log
- summary.md에는 preprocessing/label mapping/resize-crop 규칙을 명시한다.

## 5) Minimal Quality Gate
- 결과를 보고할 때는 반드시 포함:
  - (1) Overall + 조건별(mIoU) 숫자
  - (2) qualitative 실패 케이스 3장 설명
  - (3) 다음 액션 1개 (실험 1개로 제한)