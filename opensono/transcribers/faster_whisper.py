"""Faster Whisper transcription backend."""

import sys

from faster_whisper import WhisperModel

from .base import Transcriber, WordTimestamp


class FasterWhisperTranscriber(Transcriber):
    """Transcriber backed by Faster Whisper (CTranslate2 Whisper)."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[list[WordTimestamp], str | None]:
        segments_iter, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            language=language,
        )

        detected_lang = info.language
        print(
            f"Detected language: {detected_lang} "
            f"(probability {info.language_probability:.2f})",
            file=sys.stderr,
        )

        words: list[WordTimestamp] = []
        for segment in segments_iter:
            if segment.words:
                for w in segment.words:
                    words.append(
                        WordTimestamp(text=w.word, start=w.start, end=w.end)
                    )

        return words, detected_lang
