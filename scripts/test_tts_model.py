"""
Supertonic TTS 모델 테스트
- 한국어 텍스트를 PCM16LE bytes로 변환
- 각 단계의 shape 및 dtype 출력
- WAV 파일로 저장
"""
import os
import struct
import wave
from scripts.onnx_tts_engine import SupertonicONNXTTS

def pcm_to_wav(pcm_bytes: bytes, sample_rate: int, output_path: str):
    """PCM16LE bytes를 WAV 파일로 변환"""
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 모노
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

print("=" * 60)
print("Supertonic TTS Model Test")
print("=" * 60)

# --- 1. TTS 엔진 초기화 ---
tts = SupertonicONNXTTS(
    onnx_dir="assets/onnx",
    use_gpu=False,  # CUDA 사용 시 True로 변경
)

# --- 2. 한국어 텍스트 합성 ---
korean_text = "안녕하세요. 저는 슈퍼토닉 음성 합성 모델입니다."
print(f"\n[Input Text] {korean_text}")

pcm_bytes = tts.synthesize(
    text=korean_text,
    lang="ko",
    voice_style_path="assets/voice_styles/F1.json",
    total_step=5,  # 빠른 테스트용 (production: 10~30)
    speed=1.05,
)

# --- 3. 결과 출력 ---
print("\n" + "=" * 60)
print("Final Result:")
print("=" * 60)
print(f"PCM16LE bytes length: {len(pcm_bytes)}")
print(f"PCM16LE dtype: int16 (little-endian)")
print(f"Duration: ~{len(pcm_bytes) / 2 / tts.sample_rate:.2f}s")
print("=" * 60)

# --- 4. 파일 저장 ---
os.makedirs("results", exist_ok=True)

# PCM 파일 저장
pcm_path = "results/test_korean.pcm"
with open(pcm_path, "wb") as f:
    f.write(pcm_bytes)
print(f"\n✓ Saved PCM file: {pcm_path}")

# WAV 파일 저장
wav_path = "results/test_korean.wav"
pcm_to_wav(pcm_bytes, tts.sample_rate, wav_path)
print(f"✓ Saved WAV file: {wav_path}")

print(f"\n재생 방법:")
print(f"  - Python: import soundfile as sf; sf.play(sf.read('{wav_path}'))")
print(f"  - CLI: ffplay {wav_path}")
print(f"  - 또는 미디어 플레이어로 {wav_path} 파일 열기")
