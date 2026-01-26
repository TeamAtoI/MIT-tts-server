"""
TTS Service
- Supertonic ONNX TTS 엔진을 사용한 음성 합성 서비스
"""

import wave
import io
import time
import os
import logging
from pathlib import Path
from typing import Tuple, Dict
from app.inference.onnx_tts_engine import SupertonicONNXTTS

logger = logging.getLogger(__name__)


def initialize_tts_engine() -> SupertonicONNXTTS:
    """TTS 엔진 초기화 (FastAPI lifespan에서 호출)"""
    # 환경변수로 GPU 사용 여부 제어 (기본값: True)
    use_gpu = os.getenv("TTS_USE_GPU", "true").lower() == "true"
    tts = SupertonicONNXTTS(
        onnx_dir="assets/onnx",
        use_gpu=use_gpu,
    )

    # Warmup: 첫 요청 지연을 방지하기 위해 더미 합성 실행
    logger.info("[STARTUP] Running warmup synthesis...")
    _ = tts.synthesize("warmup", "en", "assets/voice_styles/F3.json")
    logger.info("[STARTUP] Warmup complete")

    return tts


def get_system_info(tts: SupertonicONNXTTS) -> Dict[str, str]:
    """시스템 및 GPU 사용 정보 반환"""
    providers = tts.dp_ort.get_providers()

    info = {
        "execution_provider": providers[0] if providers else "Unknown",
        "available_providers": ", ".join(providers),
        "device": "GPU" if "CUDA" in providers[0] else "CPU",
        "sample_rate": tts.sample_rate,
    }

    # GPU 메모리 정보 (nvidia-smi 사용)
    if info["device"] == "GPU":
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                memory = result.stdout.strip().split(",")
                info["gpu_memory"] = f"{memory[0].strip()}MB / {memory[1].strip()}MB"
        except:
            pass

    return info


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """PCM16LE bytes를 WAV bytes로 변환"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # 모노
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def synthesize(
    tts: SupertonicONNXTTS,
    text: str,
    voice: str = "F1",
    lang: str = "ko",
    total_step: int = 5,
    speed: float = 1.05,
    output_format: str = "wav",
) -> Tuple[bytes, Dict[str, float]]:
    """
    텍스트를 음성으로 합성

    Args:
        tts: TTS 엔진 인스턴스 (app.state.tts_engine)
        text: 합성할 텍스트
        voice: 음성 스타일 (F1, F2, F3, F4, F5, M1, M2, M3, M4, M5)
        lang: 언어 코드 (ko, en, es, pt, fr)
        total_step: denoising step 수 (낮을수록 빠르지만 품질 저하)
        speed: 속도 (1.0=기본, 높을수록 빠름)
        output_format: 출력 형식 (wav 또는 pcm)

    Returns:
        (오디오 bytes, 성능 메트릭)
    """
    # Voice style 파일 경로
    voice_style_path = f"assets/voice_styles/{voice}.json"
    if not Path(voice_style_path).exists():
        raise ValueError(f"Voice style not found: {voice}")

    # 레이턴시 측정 시작
    start_time = time.time()

    # PCM 합성
    logger.info(f"[TTS] 음성 합성 시작 - 텍스트: '{text[:30]}...', 음성: {voice}")
    pcm_bytes = tts.synthesize(
        text=text,
        lang=lang,
        voice_style_path=voice_style_path,
        total_step=total_step,
        speed=speed,
    )
    logger.info(f"[TTS] 음성 합성 완료 - 소요 시간: {time.time() - start_time:.3f}s")

    inference_time = time.time() - start_time

    # 출력 형식에 따라 변환
    conversion_start = time.time()
    if output_format == "wav":
        audio_bytes = pcm_to_wav_bytes(pcm_bytes, tts.sample_rate)
    else:  # pcm
        audio_bytes = pcm_bytes
    conversion_time = time.time() - conversion_start

    # 오디오 길이 계산 (초)
    audio_duration = len(pcm_bytes) / (tts.sample_rate * 2)  # 16-bit = 2 bytes

    metrics = {
        "inference_time_ms": round(inference_time * 1000, 2),
        "conversion_time_ms": round(conversion_time * 1000, 2),
        "total_time_ms": round((inference_time + conversion_time) * 1000, 2),
        "audio_duration_sec": round(audio_duration, 2),
        "rtf": round(inference_time / audio_duration, 2)
        if audio_duration > 0
        else 0,  # Real-time Factor
    }

    return audio_bytes, metrics
