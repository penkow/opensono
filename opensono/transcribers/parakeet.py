"""NVIDIA Parakeet CTC transcription backend (NeMo)."""

import gc
import os
import sys
import tempfile

import soundfile as sf

from .base import Transcriber, WordTimestamp


class ParakeetTranscriber(Transcriber):
    """Transcriber backed by NVIDIA Parakeet CTC (English-only).

    Long audio is split into ``chunk_len_s``-second segments and transcribed
    chunk-by-chunk, with word timestamps offset back to the original timeline.
    This keeps peak VRAM bounded regardless of file length.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b",
        device: str = "cuda",
        chunk_len_s: float = 300.0,
    ):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.eval()
        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        self.chunk_len_s = float(chunk_len_s)

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[list[WordTimestamp], str | None]:
        if language and language.lower() != "en":
            print(
                f"Warning: Parakeet is English-only; ignoring language={language!r}",
                file=sys.stderr,
            )

        audio, sr = sf.read(audio_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration_s = len(audio) / sr

        if duration_s <= self.chunk_len_s:
            return self._transcribe_path(audio_path), "en"

        print(
            f"Audio is {duration_s:.0f}s — chunking into "
            f"{self.chunk_len_s:.0f}s segments",
            file=sys.stderr,
        )

        chunk_size = int(self.chunk_len_s * sr)
        n_chunks = (len(audio) + chunk_size - 1) // chunk_size
        all_words: list[WordTimestamp] = []

        for idx, start_sample in enumerate(range(0, len(audio), chunk_size), 1):
            end_sample = min(start_sample + chunk_size, len(audio))
            offset_s = start_sample / sr
            print(
                f"  chunk {idx}/{n_chunks} "
                f"({offset_s:.0f}s–{end_sample / sr:.0f}s)",
                file=sys.stderr,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, audio[start_sample:end_sample], sr)
                chunk_words = self._transcribe_path(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            for w in chunk_words:
                all_words.append(
                    WordTimestamp(
                        text=w.text,
                        start=w.start + offset_s,
                        end=w.end + offset_s,
                    )
                )

            self._free_gpu_memory()

        return all_words, "en"

    def _transcribe_path(self, audio_path: str) -> list[WordTimestamp]:
        output = self.model.transcribe([audio_path], timestamps=True)
        if not output:
            return []

        hypothesis = output[0]
        word_entries = (getattr(hypothesis, "timestamp", None) or {}).get("word", [])

        words: list[WordTimestamp] = []
        for entry in word_entries:
            start = entry.get("start")
            end = entry.get("end")
            if start is None or end is None:
                stride = self._frame_stride_seconds()
                start = entry["start_offset"] * stride
                end = entry["end_offset"] * stride
            words.append(
                WordTimestamp(
                    text=entry["word"],
                    start=float(start),
                    end=float(end),
                )
            )
        return words

    def _frame_stride_seconds(self) -> float:
        """Encoder frame stride in seconds (window_stride × subsampling factor)."""
        cfg = self.model.cfg
        window_stride = cfg.preprocessor.window_stride
        subsampling = getattr(cfg.encoder, "subsampling_factor", 8)
        return float(window_stride) * float(subsampling)

    def _free_gpu_memory(self) -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
