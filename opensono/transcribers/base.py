"""Abstract transcription backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WordTimestamp:
    text: str
    start: float
    end: float
    speaker_id: int = 0


class Transcriber(ABC):
    """Abstract transcription backend producing word-level timestamps."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[list[WordTimestamp], str | None]:
        """Transcribe audio.

        Returns (words, detected_language). detected_language is None if the
        backend does not perform language detection.
        """
