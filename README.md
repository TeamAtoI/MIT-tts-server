# Supertonic TTS Server

Supertonic 2 ONNX 기반 다국어 음성 합성 (TTS) 서버

## 빠른 시작

> 📋 **전제조건**: 이 저장소를 클론했거나 소스 코드를 다운로드한 상태여야 합니다.

### 1. 모델 다운로드 (필수)

**uv 먼저 설치** (아래 2번 참고):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**모델 다운로드**:
```bash
# Hugging Face Hub을 통해 Supertonic 2 모델 다운로드
make download
```

> 📦 약 250MB의 ONNX 모델 파일이 다운로드됩니다.  
> ⚠️ 모델 없이는 서버가 실행되지 않습니다.  


### 2. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3. 의존성 설치

```bash
# GPU 버전
uv sync --extra gpu

# CPU 전용 (GPU 없는 환경)
uv sync --extra cpu
```

자세한 설치 방법은 [SETUP.md](SETUP.md)를 참고하세요.

### 4. 테스트

```bash
make test
ls results/  # test_korean.wav 확인
```

### 5. 서버 실행

```bash
# GPU 모드 (기본)
make server

# CPU 모드
make server-cpu
```

<details>
<summary>셸 스크립트 사용 (대안)</summary>

```bash
# CPU 모드
./start_cpu.sh

# GPU 모드  
./start_gpu.sh
```
</details>

GPU 설정 방법은 [SETUP.md](SETUP.md#gpu-설정)을 참고하세요.

### 6. 웹 테스트

서버가 실행된 상태에서 브라우저로 접속:

- **테스트 UI**: http://localhost:8000/example/tts_player.html
- **API 문서**: http://localhost:8000/docs
- **시스템 정보**: http://localhost:8000/tts/system

> **💡 터미널 사용법**
> 
> **단일 터미널** (권장)
> ```bash
> ./start_gpu.sh  # 서버 실행 (Ctrl+C로 종료)
> ```
> 
> 다른 터미널에서 브라우저를 열거나, 백그라운드 실행:
> ```bash
> ./start_gpu.sh > server.log 2>&1 &  # 백그라운드 실행
> tail -f server.log                   # 로그 확인
> pkill -f uvicorn                     # 종료
> ```

---

## 주요 기능

- ONNX Runtime 기반 경량 추론 (PyTorch 불필요)
- 다국어 지원 (한국어, 영어, 스페인어, 포르투갈어, 프랑스어)
- 10가지 음성 (여성 5종, 남성 5종)
- GPU/CPU 자동 전환
- REST API 및 웹 UI

---

## Docker 배포 (Compose VM + GHCR)

### 1. GHCR 로그인

```bash
export GHCR_USER=<your-github-id>
export GHCR_TOKEN=<your-ghcr-token>
make docker-login-ghcr
```

### 2. 이미지 빌드 (linux/amd64)

```bash
# GPU 이미지
make docker-build-gpu TAG=v0.1.0

# CPU 이미지
make docker-build-cpu TAG=v0.1.0
```

기본 모델 리비전:

- `75e6727618a02f323c720cba9478152d4bc16ca4`

필요 시 다른 리비전으로 빌드:

```bash
make docker-build-gpu TAG=v0.1.0 MODEL_REVISION=<revision-sha>
```

### 3. GHCR 푸시

```bash
# GPU/CPU 모두 푸시 + latest 이동 태그 갱신
make docker-push-all TAG=v0.1.0
```

푸시되는 태그:

- `ghcr.io/teamatoi/mit-tts:gpu-v0.1.0`
- `ghcr.io/teamatoi/mit-tts:cpu-v0.1.0`
- `ghcr.io/teamatoi/mit-tts:gpu-latest`
- `ghcr.io/teamatoi/mit-tts:cpu-latest`

### 4. GPU 서버 실행 (Docker Compose)

```bash
# 기본 이미지: ghcr.io/teamatoi/mit-tts:gpu-latest
docker compose pull
docker compose up -d

# 특정 태그로 오버라이드
TTS_GPU_IMAGE=ghcr.io/teamatoi/mit-tts:gpu-v0.1.0 docker compose up -d
```

접속:

- `http://<server-ip>:8005/health`
- `http://<server-ip>:8005/docs`

중지:

```bash
docker compose down
```

### 5. CPU 이미지 단독 실행 (docker run)

```bash
docker run -d \
  --name tts-cpu \
  -p 8005:8000 \
  -e TTS_USE_GPU=false \
  ghcr.io/teamatoi/mit-tts:cpu-latest
```

---

## 문서

- [IMPLEMENTATION.md](IMPLEMENTATION.md) - 아키텍처, 모델 설명, API 상세
- [SETUP.md](SETUP.md) - 환경 설정, 트러블슈팅
- [Swagger UI](http://localhost:8000/docs) - API 대화형 문서 (서버 실행 후)

---

## API 엔드포인트

### POST /tts/synthesize

```bash
curl -X POST http://localhost:8000/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "안녕하세요", "voice": "F1"}' \
  --output output.wav
```

### GET /tts/voices

사용 가능한 음성 목록

### GET /health

서버 상태 확인

상세한 API 명세는 서버 실행 후 http://localhost:8000/docs 참고하세요.

---

## 라이센스

서버 코드: MIT License  
Supertonic 모델: Supertone Inc. 라이센스 참조

- [Supertonic GitHub](https://github.com/supertone-inc/supertonic)
- [ONNX Runtime](https://onnxruntime.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
