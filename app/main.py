# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import tts
from app.service.tts_service import initialize_tts_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션 생명주기 관리
    - startup: TTS 엔진을 미리 로드하여 첫 요청 지연 방지
    - shutdown: 리소스 정리
    """
    # Startup: TTS 엔진 초기화
    print("[STARTUP] Loading TTS engine...")
    app.state.tts_engine = initialize_tts_engine()
    print("[STARTUP] TTS engine loaded successfully")

    yield


app = FastAPI(title="Supertonic TTS Server", version="0.1.0", lifespan=lifespan)

# CORS 설정 (웹 UI에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Inference-Time-Ms",
        "X-Total-Time-Ms",
        "X-Audio-Duration-Sec",
        "X-RTF",
    ],
)


app.include_router(tts.router)

# 정적 파일 서빙 (example 폴더의 HTML 등)
app.mount("/example", StaticFiles(directory="example"), name="example")


@app.get("/")
def root():
    return {
        "message": "Supertonic TTS Server",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "tts": "/tts/synthesize (POST)",
            "voices": "/tts/voices",
            "system": "/tts/system",
            "test_ui": "/example/tts_player.html",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
