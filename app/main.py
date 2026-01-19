# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes import tts

app = FastAPI(title="Supertonic TTS Server", version="0.1.0")

# CORS 설정 (웹 UI에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static 파일 서빙 (테스트 UI)
app.mount("/tests", StaticFiles(directory="tests"), name="tests")

app.include_router(tts.router)

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
            "test_ui": "/tests/tts_player.html"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
