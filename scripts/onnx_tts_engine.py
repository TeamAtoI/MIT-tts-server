"""
Supertonic ONNX TTS Engine
- 4개의 ONNX 모델을 사용한 전체 파이프라인
- Text → Duration → Text Embedding → Denoising → Vocoder
"""
import json
import os
import re
from typing import Optional
from unicodedata import normalize

import numpy as np
import onnxruntime as ort


AVAILABLE_LANGS = ["en", "ko", "es", "pt", "fr"]


class UnicodeProcessor:
    """텍스트를 유니코드 인덱스로 변환"""
    
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        """텍스트 전처리 및 언어 태그 추가"""
        text = normalize("NFKD", text)
        
        # 이모지 제거
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f"
            "\U0001f780-\U0001f7ff"
            "\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff"
            "\U0001fa00-\U0001fa6f"
            "\U0001fa70-\U0001faff"
            "\u2600-\u26ff"
            "\u2700-\u27bf"
            "\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        
        # 특수문자 치환
        replacements = {
            "–": "-", "‑": "-", "—": "-", "_": " ",
            "\u201c": '"', "\u201d": '"',
            "\u2018": "'", "\u2019": "'",
            "´": "'", "`": "'",
            "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
            "→": " ", "←": " ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        text = re.sub(r"[♥☆♡©\\]", "", text)
        
        # 축약어
        expr_replacements = {
            "@": " at ",
            "e.g.,": "for example, ",
            "i.e.,": "that is, ",
        }
        for k, v in expr_replacements.items():
            text = text.replace(k, v)
        
        # 문장부호 앞 공백 제거
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)
        
        # 중복 따옴표 제거
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        
        text = re.sub(r"\s+", " ", text).strip()
        
        # 마침표 추가 (없으면)
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."
        
        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        
        text = f"<{lang}>" + text + f"</{lang}>"
        return text
    
    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        return np.array([ord(char) for char in text], dtype=np.uint16)
    
    def __call__(self, text_list: list[str], lang_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """텍스트 리스트를 인덱스와 마스크로 변환"""
        text_list = [self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, :len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask
    
    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        """길이 배열을 마스크로 변환 (B, 1, max_len)"""
        max_len = text_ids_lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(text_ids_lengths, axis=1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)


class Style:
    """Voice style을 담는 컨테이너"""
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class SupertonicONNXTTS:
    """Supertonic 2 ONNX TTS 엔진"""
    
    def __init__(self, onnx_dir: str, use_gpu: bool = False):
        """
        Args:
            onnx_dir: ONNX 모델들이 있는 디렉토리
            use_gpu: CUDA 사용 여부 (실패 시 CPU fallback)
        """
        self.onnx_dir = onnx_dir
        
        # --- Provider 설정 ---
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[INFO] Attempting to use GPU (CUDA)")
        else:
            providers = ["CPUExecutionProvider"]
            print("[INFO] Using CPU for inference")
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # --- ONNX 모델 로드 ---
        self.dp_ort = ort.InferenceSession(
            os.path.join(onnx_dir, "duration_predictor.onnx"),
            sess_options=opts,
            providers=providers,
        )
        self.text_enc_ort = ort.InferenceSession(
            os.path.join(onnx_dir, "text_encoder.onnx"),
            sess_options=opts,
            providers=providers,
        )
        self.vector_est_ort = ort.InferenceSession(
            os.path.join(onnx_dir, "vector_estimator.onnx"),
            sess_options=opts,
            providers=providers,
        )
        self.vocoder_ort = ort.InferenceSession(
            os.path.join(onnx_dir, "vocoder.onnx"),
            sess_options=opts,
            providers=providers,
        )
        
        # --- Config 로드 ---
        with open(os.path.join(onnx_dir, "tts.json"), "r") as f:
            self.cfgs = json.load(f)
        
        self.sample_rate = self.cfgs["ae"]["sample_rate"]
        self.base_chunk_size = self.cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = self.cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = self.cfgs["ttl"]["latent_dim"]
        
        # --- Text Processor 로드 ---
        unicode_indexer_path = os.path.join(onnx_dir, "unicode_indexer.json")
        self.text_processor = UnicodeProcessor(unicode_indexer_path)
        
        print(f"[OK] Supertonic ONNX TTS Engine initialized")
        print(f"     Providers: {self.dp_ort.get_providers()}")
        print(f"     Sample rate: {self.sample_rate} Hz")
    
    def load_voice_style(self, voice_style_paths: list[str]) -> Style:
        """
        Voice style JSON 파일 로드
        
        Args:
            voice_style_paths: voice style JSON 경로 리스트
        
        Returns:
            Style 객체
        """
        bsz = len(voice_style_paths)
        
        # 첫 파일로 dimension 파악
        with open(voice_style_paths[0], "r") as f:
            first_style = json.load(f)
        
        ttl_dims = first_style["style_ttl"]["dims"]
        dp_dims = first_style["style_dp"]["dims"]
        
        # 배열 할당
        ttl_style = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
        dp_style = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)
        
        # 데이터 채우기
        for i, path in enumerate(voice_style_paths):
            with open(path, "r") as f:
                voice_style = json.load(f)
            
            ttl_data = np.array(voice_style["style_ttl"]["data"], dtype=np.float32).flatten()
            ttl_style[i] = ttl_data.reshape(ttl_dims[1], ttl_dims[2])
            
            dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
            dp_style[i] = dp_data.reshape(dp_dims[1], dp_dims[2])
        
        return Style(ttl_style, dp_style)
    
    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Duration에 맞춰 noisy latent 생성"""
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        
        # Latent mask 생성
        latent_size = self.base_chunk_size * self.chunk_compress_factor
        latent_lengths = (wav_lengths + latent_size - 1) // latent_size
        latent_mask = self._length_to_mask(latent_lengths)
        
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask
    
    def _length_to_mask(self, lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
        """길이 배열을 마스크로 변환 (B, 1, max_len)"""
        max_len = max_len or lengths.max()
        ids = np.arange(0, max_len)
        mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
        return mask.reshape(-1, 1, max_len)
    
    def synthesize(
        self,
        text: str,
        lang: str,
        voice_style_path: str,
        total_step: int = 5,
        speed: float = 1.05,
    ) -> bytes:
        """
        한국어 텍스트를 PCM16LE bytes로 변환
        
        Args:
            text: 입력 텍스트
            lang: 언어 코드 (ko, en, es, pt, fr)
            voice_style_path: voice style JSON 경로
            total_step: denoising step 수 (기본 5)
            speed: 속도 (기본 1, 높을수록 빠름)
        
        Returns:
            PCM16LE bytes (int16 little-endian)
        """
        # --- 1. Voice style 로드 ---
        style = self.load_voice_style([voice_style_path])
        
        # --- 2. 텍스트를 인덱스로 변환 ---
        text_ids, text_mask = self.text_processor([text], [lang])
        print(f"[INFO] Text IDs shape: {text_ids.shape}")
        print(f"[INFO] Text mask shape: {text_mask.shape}")
        
        # --- 3. Duration Prediction ---
        dur_onnx, *_ = self.dp_ort.run(
            None,
            {
                "text_ids": text_ids,
                "style_dp": style.dp,
                "text_mask": text_mask,
            }
        )
        dur_onnx = dur_onnx / speed
        print(f"[INFO] Duration: {dur_onnx[0]:.2f}s (speed={speed})")
        
        # --- 4. Text Encoding ---
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {
                "text_ids": text_ids,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
            }
        )
        print(f"[INFO] Text embedding shape: {text_emb_onnx.shape}")
        
        # --- 5. Denoising (Vector Estimation) ---
        xt, latent_mask = self._sample_noisy_latent(dur_onnx)
        print(f"[INFO] Initial noisy latent shape: {xt.shape}")
        
        bsz = len(text_ids)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                }
            )
        
        print(f"[INFO] Denoised latent shape: {xt.shape}")
        
        # --- 6. Vocoding ---
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        print(f"[INFO] Waveform shape: {wav.shape}, dtype: {wav.dtype}")
        
        # --- 7. Trim to duration ---
        wav_trimmed = wav[0, :int(self.sample_rate * dur_onnx[0])]
        print(f"[INFO] Trimmed waveform shape: {wav_trimmed.shape}")
        
        # --- 8. Convert to PCM16LE bytes ---
        wav_int16 = (wav_trimmed * 32767.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = wav_int16.tobytes()
        
        print(f"[INFO] PCM16LE bytes length: {len(pcm_bytes)}")
        print(f"[INFO] PCM16LE dtype: int16 (little-endian)")
        
        return pcm_bytes
