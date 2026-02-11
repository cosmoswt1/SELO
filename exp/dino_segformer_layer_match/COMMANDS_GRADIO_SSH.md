# DINO-SegFormer Gradio + SSH 명령어 모음

이 파일은 `exp/dino_segformer_layer_match`의 Gradio 클릭 UI를
서버에서 실행하고, 맥에서 SSH 포트포워딩으로 접속할 때 쓰는
고정 명령어 모음이다.

## 0) 서버에서 먼저 확인 (GPU 필수)

```bash
cd ~/SELO
nvidia-smi
conda run -n selo python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

- `torch.cuda.is_available()`가 `True`일 때만 진행.
- `False`면 GPU 상태 복구 후 재실행.

## 1) 서버에서 Gradio 실행

기본(권장: 원격 로컬 바인딩 `127.0.0.1`):

```bash
cd ~/SELO
HOST=127.0.0.1 PORT=7860 CONDITION=night \
OUT_DIR=~/SELO/exp/dino_segformer_layer_match/gradio_picker \
bash exp/dino_segformer_layer_match/run_gradio_anchor.sh
```

패널 더 크게 보고 싶을 때:

```bash
cd ~/SELO
HOST=127.0.0.1 PORT=7860 CONDITION=night \
OUT_DIR=~/SELO/exp/dino_segformer_layer_match/gradio_picker \
bash exp/dino_segformer_layer_match/run_gradio_anchor.sh \
  --input_display_height 1050 \
  --panel_display_height 1900 \
  --panel_cell_inches 5.0 \
  --panel_dpi 260
```

## 2) 맥에서 SSH 포트포워딩

### 실사용 예시 (현재 환경)

```bash
ssh -i ~/.ssh/id_rsa -p 11462 -N -L 7860:127.0.0.1:7860 kevinlee01@192.168.45.11
```

### 템플릿

```bash
ssh -i <key_path> -p <ssh_port> -N -L <local_port>:127.0.0.1:<remote_port> <user>@<server_ip>
```

- 보통 `<local_port>`와 `<remote_port>`를 둘 다 `7860`으로 맞춤.
- 충돌 시 로컬 포트만 변경 가능: `-L 17860:127.0.0.1:7860`

## 3) 맥 브라우저 접속

```text
http://localhost:7860
```

로컬 포트를 17860으로 포워딩했다면:

```text
http://localhost:17860
```

## 4) 빠른 상태 점검

서버에서(Gradio 살아있는지):

```bash
curl -I http://127.0.0.1:7860
ps -ef | rg "gradio_anchor_app.py|run_gradio_anchor.sh"
```

맥에서(터널 프로세스 확인):

```bash
ps -ef | rg "ssh .* -L 7860:127.0.0.1:7860"
```

## 5) 자주 헷갈리는 포인트

- `SSH_PORT_HINT`는 **안내 문구 출력용**이며 실제 SSH 접속에 영향 없음.
- 실제 접속은 맥에서 실행한 `ssh -p ... -L ...` 명령이 결정함.
- 브라우저 접속 실패 시 우선 서버에서 `curl -I http://127.0.0.1:7860`이 되는지 확인.

