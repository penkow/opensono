"""NVIDIA Parakeet CTC transcription backend (NeMo)."""

import gc
import os
import sys
import tempfile

import soundfile as sf

from .base import Transcriber, WordTimestamp


class ParakeetTranscriber(Transcriber):
    """Transcriber backed by NVIDIA Parakeet CTC (English-only).

    For audio longer than ``chunk_len_s``, the waveform is split into
    non-overlapping segments, each transcribed independently with word-level
    timestamps that are offset back to the global timeline. This is reliable
    and produces clean word timestamps suitable for downstream diarization.

    An experimental ``native_chunking=True`` path routes long audio through
    NeMo's :class:`FrameBatchASR` instead. Whether it yields usable word
    timestamps depends on the installed NeMo version's decoding plumbing — if
    none come back, this path returns an empty word list and prints a warning
    rather than fabricating timestamps.
    """

    # Hardcoded for the experimental native path. FrameBatchASR is designed
    # for streaming-style frames in the tens-of-seconds range — not the
    # 5-minute "chunk_len_s" used by manual chunking.
    _NATIVE_FRAME_LEN_S = 30.0
    _NATIVE_CONTEXT_S = 4.0

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-ctc-0.6b",
        device: str = "cuda",
        chunk_len_s: float = 300.0,
        native_chunking: bool = False,
    ):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.model.eval()
        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        self.chunk_len_s = float(chunk_len_s)
        self.native_chunking = native_chunking

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

        if self.native_chunking:
            return self._transcribe_native_chunked(audio_path, duration_s), "en"
        return self._transcribe_manual_chunked(audio, sr), "en"

    # ---------------------------------------------------------------- manual

    def _transcribe_manual_chunked(self, audio, sr) -> list[WordTimestamp]:
        print(
            f"Audio is {len(audio) / sr:.0f}s — chunking into "
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

        return all_words

    # ----------------------------------------------------- native (FrameBatchASR)

    def _transcribe_native_chunked(
        self,
        audio_path: str,
        duration_s: float,
    ) -> list[WordTimestamp]:
        from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
        from omegaconf import OmegaConf

        frame_len_s = self._NATIVE_FRAME_LEN_S
        context_s = self._NATIVE_CONTEXT_S
        stride = self._frame_stride_seconds()

        print(
            f"Audio is {duration_s:.0f}s — FrameBatchASR "
            f"(frame={frame_len_s:.0f}s, context={context_s:.0f}s)",
            file=sys.stderr,
        )

        try:
            decoding_cfg = OmegaConf.to_container(
                self.model.cfg.decoding, resolve=True
            )
            decoding_cfg["compute_timestamps"] = True
            decoding_cfg["preserve_alignments"] = True
            self.model.change_decoding_strategy(OmegaConf.create(decoding_cfg))
        except Exception as e:
            print(
                f"Warning: could not enable timestamps in decoding: {e}",
                file=sys.stderr,
            )

        frame_asr = FrameBatchASR(
            asr_model=self.model,
            frame_len=frame_len_s,
            total_buffer=frame_len_s + 2 * context_s,
            batch_size=1,
        )
        frame_asr.reset()
        frame_asr.read_audio_file(audio_path, delay=0.0, model_stride_in_secs=stride)

        tokens_per_chunk = int(frame_len_s / stride)
        mid_delay = int(context_s / stride)
        hyp = frame_asr.transcribe(tokens_per_chunk=tokens_per_chunk, delay=mid_delay)

        words = self._extract_words(hyp)
        self._free_gpu_memory()

        if not words:
            print(
                "Warning: FrameBatchASR returned no word-level timestamps in this "
                "NeMo build. Re-run without --native-chunking to use manual "
                "waveform chunking, which produces precise word timing.",
                file=sys.stderr,
            )
        return words

    # ---------------------------------------------------------------- shared

    def _transcribe_path(self, audio_path: str) -> list[WordTimestamp]:
        output = self.model.transcribe([audio_path], timestamps=True)
        if not output:
            return []
        return self._extract_words(output[0])

    def _extract_words(self, hyp) -> list[WordTimestamp]:
        """Pull a list[WordTimestamp] out of a NeMo Hypothesis, if present."""
        if isinstance(hyp, str) or hyp is None:
            return []
        ts = getattr(hyp, "timestamp", None)
        if not ts:
            return []
        word_entries = ts.get("word", []) if isinstance(ts, dict) else []
        out: list[WordTimestamp] = []
        for entry in word_entries:
            start = entry.get("start")
            end = entry.get("end")
            if start is None or end is None:
                stride = self._frame_stride_seconds()
                start = entry.get("start_offset", 0) * stride
                end = entry.get("end_offset", 0) * stride
            out.append(
                WordTimestamp(
                    text=entry.get("word", ""),
                    start=float(start),
                    end=float(end),
                )
            )
        return out

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
