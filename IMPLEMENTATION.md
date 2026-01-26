# Supertonic TTS Server - 구현 가이드

Supertonic 2 TTS 서버의 아키텍처, 모델 설명, API 상세 문서입니다.

## 목차

- [시스템 아키텍처](#시스템-아키텍처)
- [모델 설명](#모델-설명)
- [API 사용법](#api-사용법)
- [기술 스택](#기술-스택)
- [재구현 가이드](#재구현-가이드)

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────┐
│          FastAPI Server (main.py)           │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   Routes (routes/tts.py)   │
    └─────────────┬───────────────┘
                  │
    ┌─────────────┴─────────────────┐
    │  Service (service/tts_service) │
    └─────────────┬───────────────────┘
                  │
    ┌─────────────┴─────────────────────────┐
    │  TTS Engine (scripts/onnx_tts_engine)  │
    └─────────────┬───────────────────────────┘
                  │
    ┌─────────────┴───────────────────┐
    │   4 ONNX Models (assets/onnx/)  │
    │   - duration_predictor.onnx      │
    │   - text_encoder.onnx           │
    │   - vector_estimator.onnx       │
    │   - vocoder.onnx                │
    └─────────────────────────────────┘
```

### 데이터 흐름

```
Text (str)
  ↓
[Text Preprocessing] → Unicode Normalization + Language Tagging
  ↓
[Text Encoder] → Text Embeddings (256-dim)
  ↓
[Duration Predictor] → Speech Duration (seconds)
  ↓
[Vector Estimator] → Denoising (5-30 steps)
  ↓
[Vocoder] → Waveform (float32)
  ↓
[PCM Conversion] → PCM16LE bytes
  ↓
[WAV Wrapper] → WAV file (optional)
```

---

## 기술 스택

### Core Dependencies

| 패키지 | 버전 | 용도 |
|--------|------|------|
| **onnxruntime** | ≥1.18.0 (CPU) | ONNX 모델 추론 엔진 |
| **onnxruntime-gpu** | ==1.16.3 (GPU) | GPU 가속 (CUDA 11.8 + cuDNN 8.x) |
| **numpy** | ≥1.24.0, <2.0 | 수치 연산 |
| **fastapi** | ≥0.115.0 | REST API 프레임워크 |
| **uvicorn** | ≥0.30.0 | ASGI 서버 |
| **pydantic** | ≥2.8.0 | 데이터 검증 |

환경 설정 및 설치 방법은 [SETUP.md](SETUP.md)를 참고하세요.

---

## 모델 설명

### Supertonic 2 아키텍처

Supertonic 2는 **Latent Diffusion 기반 TTS 모델**로, 4단계 파이프라인으로 구성됩니다.

#### 1. Duration Predictor

- **입력**: 텍스트 ID (int64), 스타일 (style_dp), 텍스트 마스크
- **출력**: 예상 음성 길이 (초)
- **역할**: 각 텍스트가 얼마나 긴 음성이 될지 예측

#### 2. Text Encoder

- **입력**: 텍스트 ID, 스타일 (style_ttl), 텍스트 마스크
- **출력**: 텍스트 임베딩 (B, 256, T)
- **역할**: 텍스트의 의미와 운율 정보를 256차원 벡터로 인코딩

#### 3. Vector Estimator (Denoiser)

- **입력**: 
  - noisy_latent (random noise)
  - text_emb
  - style_ttl
  - current_step, total_step
- **출력**: 디노이즈된 latent (B, 144, L)
- **역할**: Diffusion 방식으로 noise에서 음성 latent를 생성 (5-30 step)

#### 4. Vocoder

- **입력**: Latent (B, 144, L)
- **출력**: Waveform (B, T_audio) @ 44.1kHz
- **역할**: Latent를 실제 오디오 파형으로 변환

### Voice Style

각 음색 JSON에는 두 가지 스타일 벡터가 포함됩니다:

- **style_ttl**: Text-to-Latent 스타일 (1, 8, 16)
- **style_dp**: Duration Predictor 스타일 (1, 4, 16)

이 벡터들이 음색의 고유 특성(톤, 억양, 발음)을 결정합니다.

### 품질 vs 속도 Trade-off

| total_step | 속도 | 품질 | 용도 |
|------------|------|------|------|
| 5 | 매우 빠름 | 보통 | 실시간 데모, 빠른 프로토타입 |
| 10 | 빠름 | 좋음 | 일반적 사용 |
| 20 | 보통 | 우수 | 프로덕션 품질 |
| 30 | 느림 | 최고 | 최고 품질 필요 시 |

---

## API 사용법

### 엔드포인트

#### `POST /tts/synthesize`

텍스트를 음성으로 합성합니다.

**Request Body**:

```json
{
  "text": "안녕하세요. 슈퍼토닉입니다.",
  "voice": "F1",
  "lang": "ko",
  "speed": 1.05,
  "total_step": 5,
  "output_format": "wav"
}
```

**Parameters**:

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| text | string | *required* | 합성할 텍스트 (최대 1000자) |
| voice | string | "F1" | F1-F5 (여성), M1-M5 (남성) |
| lang | string | "ko" | ko, en, es, pt, fr |
| speed | float | 1.05 | 0.5 ~ 2.0 (낮을수록 느림) |
| total_step | int | 5 | 1 ~ 30 (높을수록 고품질) |
| output_format | string | "wav" | wav 또는 pcm |

**Response**: 
- Content-Type: `audio/wav` 또는 `audio/pcm`
- Binary audio data

#### `GET /tts/voices`

사용 가능한 음성 목록을 반환합니다.

**Response**:

```json
{
  "voices": {
    "female": ["F1", "F2", "F3", "F4", "F5"],
    "male": ["M1", "M2", "M3", "M4", "M5"]
  },
  "languages": ["ko", "en", "es", "pt", "fr"]
}
```

#### `GET /health`

서버 상태를 확인합니다.

**Response**:

```json
{
  "status": "ok"
}
```

### Python 클라이언트 예제

```python
import requests

# 음성 합성
response = requests.post(
    "http://localhost:8000/tts/synthesize",
    json={
        "text": "안녕하세요",
        "voice": "F1",
        "lang": "ko",
        "speed": 1.05,
        "total_step": 10
    }
)

# WAV 파일로 저장
with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL 예제

```bash
curl -X POST http://localhost:8000/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is Supertonic",
    "voice": "M1",
    "lang": "en",
    "speed": 1.2,
  }' \
  --output hello.wav
```

---

## 성능 메트릭

서버는 HTTP 응답 헤더로 성능 메트릭을 제공합니다:

- `X-Inference-Time`: 모델 추론 시간 (초)
- `X-Total-Time`: 전체 처리 시간 (초)
- `X-Audio-Duration`: 생성된 오디오 길이 (초)
- `X-RTF`: Real-Time Factor (추론시간 / 오디오길이)

### RTF (Real-Time Factor) 해석

- RTF < 1.0: 실시간보다 빠름 (예: 0.5 = 2배 빠름)
- RTF = 1.0: 실시간
- RTF > 1.0: 실시간보다 느림

---

## 라이센스

서버 코드: MIT License  
Supertonic 모델: Supertone Inc. 라이센스 참조
