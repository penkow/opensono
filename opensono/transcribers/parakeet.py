"""NVIDIA Parakeet CTC transcription backend (NeMo)."""

import sys

from .base import Transcriber, WordTimestamp


class ParakeetTranscriber(Transcriber):
    """Transcriber backed by NVIDIA Parakeet CTC (English-only)."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b",
        device: str = "cuda",
    ):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.eval()
        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

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

        output = self.model.transcribe([audio_path], timestamps=True)
        if not output:
            return [], "en"

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

        return words, "en"

    def _frame_stride_seconds(self) -> float:
        """Encoder frame stride in seconds (window_stride × subsampling factor)."""
        cfg = self.model.cfg
        window_stride = cfg.preprocessor.window_stride
        subsampling = getattr(cfg.encoder, "subsampling_factor", 8)
        return float(window_stride) * float(subsampling)
