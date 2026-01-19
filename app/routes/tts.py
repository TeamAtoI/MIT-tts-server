"""
TTS API Routes
"""
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel, Field
from app.service.tts_service import synthesize, get_system_info

router = APIRouter(prefix="/tts", tags=["TTS"])

class TTSRequest(BaseModel):
    text: str = Field(..., description="합성할 텍스트", min_length=1, max_length=1000)
    voice: str = Field(default="F1", description="음성 스타일 (F1-F5, M1-M5)")
    lang: str = Field(default="ko", description="언어 (ko, en, es, pt, fr)")
    speed: float = Field(default=1.05, description="속도 (0.5 ~ 2.0)", ge=0.5, le=2.0)
    total_step: int = Field(default=5, description="품질 (5=빠름, 30=고품질)", ge=1, le=30)
    output_format: str = Field(default="pcm", description="출력 형식 (wav 또는 pcm)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 슈퍼토닉 음성 합성입니다.",
                "voice": "F1",
                "lang": "ko",
                "speed": 1.05,
                "total_step": 5,
                "output_format": "wav"
            }
        }

@router.post("/synthesize", response_class=Response)
async def synthesize_speech(req: TTSRequest):
    """
    텍스트를 음성으로 합성
    
    - **text**: 합성할 텍스트 (최대 1000자)
    - **voice**: F1, F2, F3, F4, F5 (여성) 또는 M1, M2, M3, M4, M5 (남성)
    - **lang**: ko (한국어), en (영어), es (스페인어), pt (포르투갈어), fr (프랑스어)
    - **speed**: 0.5 (느림) ~ 2.0 (빠름), 기본 1.05
    - **total_step**: 5 (빠름/저품질) ~ 30 (느림/고품질), 기본 5
    - **output_format**: wav (WAV 파일) 또는 pcm (raw PCM)
    """
    try:
        audio_bytes, metrics = synthesize(
            text=req.text,
            voice=req.voice,
            lang=req.lang,
            speed=req.speed,
            total_step=req.total_step,
            output_format=req.output_format,
        )
        
        media_type = "audio/wav" if req.output_format == "wav" else "audio/pcm"
        
        # 성능 메트릭을 헤더로 전달
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Inference-Time-Ms": str(metrics["inference_time_ms"]),
                "X-Total-Time-Ms": str(metrics["total_time_ms"]),
                "X-Audio-Duration-Sec": str(metrics["audio_duration_sec"]),
                "X-RTF": str(metrics["rtf"]),
                "Cache-Control": "no-cache"
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 합성 실패: {str(e)}")

@router.get("/voices")
async def list_voices():
    """사용 가능한 음성 목록"""
    return {
        "voices": {
            "female": ["F1", "F2", "F3", "F4", "F5"],
            "male": ["M1", "M2", "M3", "M4", "M5"]
        },
        "languages": ["ko", "en", "es", "pt", "fr"]
    }

@router.get("/system")
async def system_info():
    """시스템 및 GPU 사용 정보"""
    return get_system_info()

@router.get("/synthesize")
async def synthesize_info():
    """
    GET 요청 안내
    
    이 엔드포인트는 POST 메서드만 지원합니다.
    테스트를 위해 /tests/tts_player.html을 사용하세요.
    """
    return {
        "message": "이 엔드포인트는 POST 메서드만 지원합니다.",
        "usage": {
            "method": "POST",
            "url": "/tts/synthesize",
            "example": {
                "text": "안녕하세요",
                "voice": "F1",
                "lang": "ko",
                "speed": 1.05,
                "total_step": 5,
                "output_format": "wav"
            }
        },
        "test_ui": "http://localhost:8000/tests/tts_player.html"
    }
