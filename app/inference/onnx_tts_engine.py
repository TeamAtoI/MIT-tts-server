"""
Supertonic ONNX TTS Engine (Simplified)
- IOBinding 없이 단순 session.run() 사용
- 성능 차이 ~1.5% (88ms vs 87ms)
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
        text = normalize("NFKD", text)

        # 이모지 제거
        emoji_pattern = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
            "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

        # 특수문자 치환
        for k, v in {
            "–": "-", "‑": "-", "—": "-", "_": " ",
            "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
            "´": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ",
            "#": " ", "→": " ", "←": " ",
        }.items():
            text = text.replace(k, v)

        text = re.sub(r"[♥☆♡©\\]", "", text)

        for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
            text = text.replace(k, v)

        # 문장부호 앞 공백 제거
        for punct in [",", ".", "!", "?", ";", ":", "'"]:
            text = text.replace(f" {punct}", punct)

        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")

        text = re.sub(r"\s+", " ", text).strip()

        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."

        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")

        return f"<{lang}>{text}</{lang}>"

    def __call__(self, text_list: list[str], lang_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t, lang) for t, lang in zip(text_list, lang_list)]
        lengths = np.array([len(t) for t in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), lengths.max()), dtype=np.int64)

        for i, text in enumerate(text_list):
            unicode_vals = np.array([ord(c) for c in text], dtype=np.uint16)
            text_ids[i, :len(unicode_vals)] = [self.indexer[v] for v in unicode_vals]

        # mask: (B, 1, max_len)
        max_len = lengths.max()
        mask = (np.arange(max_len) < lengths[:, None]).astype(np.float32).reshape(-1, 1, max_len)
        return text_ids, mask


class Style:
    """Voice style 컨테이너"""
    def __init__(self, ttl: np.ndarray, dp: np.ndarray):
        self.ttl = ttl
        self.dp = dp


class SupertonicONNXTTS:
    """Supertonic ONNX TTS 엔진 (Simplified)"""

    def __init__(self, onnx_dir: str, use_gpu: bool = False):
        if use_gpu:
            # V100 등 구형 GPU에서 cuDNN 9.x 호환성을 위한 설정
            cuda_provider_options = {
                "cudnn_conv_algo_search": "DEFAULT",  # EXHAUSTIVE 대신 DEFAULT 사용
                "cudnn_conv_use_max_workspace": "0",
            }
            providers = [
                ("CUDAExecutionProvider", cuda_provider_options),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.dp_ort = ort.InferenceSession(f"{onnx_dir}/duration_predictor.onnx", opts, providers)
        self.text_enc_ort = ort.InferenceSession(f"{onnx_dir}/text_encoder.onnx", opts, providers)
        self.vector_est_ort = ort.InferenceSession(f"{onnx_dir}/vector_estimator.onnx", opts, providers)
        self.vocoder_ort = ort.InferenceSession(f"{onnx_dir}/vocoder.onnx", opts, providers)

        with open(f"{onnx_dir}/tts.json", "r") as f:
            cfg = json.load(f)

        self.sample_rate = cfg["ae"]["sample_rate"]
        self.base_chunk_size = cfg["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfg["ttl"]["chunk_compress_factor"]
        self.ldim = cfg["ttl"]["latent_dim"]

        self.text_processor = UnicodeProcessor(f"{onnx_dir}/unicode_indexer.json")

    def load_voice_style(self, path: str) -> Style:
        with open(path, "r") as f:
            data = json.load(f)

        ttl = np.array(data["style_ttl"]["data"], dtype=np.float32).reshape(1, *data["style_ttl"]["dims"][1:])
        dp = np.array(data["style_dp"]["data"], dtype=np.float32).reshape(1, *data["style_dp"]["dims"][1:])
        return Style(ttl, dp)

    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_dim = self.ldim * self.chunk_compress_factor

        wav_len_max = duration.max() * self.sample_rate
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)

        noisy = np.random.randn(1, latent_dim, latent_len).astype(np.float32)

        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
        mask = (np.arange(latent_len) < latent_lengths[:, None]).astype(np.float32).reshape(1, 1, -1)

        return noisy * mask, mask

    def synthesize(
        self,
        text: str,
        lang: str,
        voice_style_path: str,
        total_step: int = 5,
        speed: float = 1.05,
    ) -> bytes:
        """텍스트를 PCM16LE bytes로 변환"""
        style = self.load_voice_style(voice_style_path)
        text_ids, text_mask = self.text_processor([text], [lang])

        # Duration Prediction
        (dur,) = self.dp_ort.run(None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask})
        dur = dur / speed

        # Text Encoding
        (text_emb,) = self.text_enc_ort.run(None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask})

        # Denoising
        xt, latent_mask = self._sample_noisy_latent(dur)
        total_step_np = np.array([total_step], dtype=np.float32)

        for step in range(total_step):
            (xt,) = self.vector_est_ort.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": np.array([step], dtype=np.float32),
                "total_step": total_step_np,
            })

        # Vocoding
        (wav,) = self.vocoder_ort.run(None, {"latent": xt})

        # Trim & Convert
        wav_trimmed = wav[0, :int(self.sample_rate * dur[0])]
        wav_int16 = (wav_trimmed * 32767.0).clip(-32768, 32767).astype(np.int16)

        return wav_int16.tobytes()
