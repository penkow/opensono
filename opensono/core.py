#!/usr/bin/env python3
"""
Transcribe and diarize audio files using Faster Whisper + NeMo Sortformer.

Produces speaker-attributed transcription with word-level timestamps.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


@dataclass
class WordTimestamp:
    text: str
    start: float
    end: float
    speaker_id: int = 0


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker_id: int


@dataclass
class TranscriptChunk:
    text: str
    start_time: float
    end_time: float
    speaker_id: int


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

def load_diarization_model():
    """Load the NeMo Sortformer diarization model."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1"
    )
    model.eval()

    # Configure for chunked / streaming-style processing
    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_update_period = 300

    return model


def diarize_audio(diar_model, audio_path: str) -> list[SpeakerSegment]:
    """Run diarization and return speaker segments."""
    predicted = diar_model.diarize(audio=[audio_path], batch_size=1)

    segments: list[SpeakerSegment] = []

    annotation = predicted[0]

    # NeMo returns a pyannote Annotation; iterating yields (segment, track, label) tuples.
    # Fall back to iterating plain segment objects or strings for other backends.
    try:
        items = list(annotation.itertracks(yield_label=True))
        for segment, _, speaker_label in items:
            start = float(segment.start)
            end = float(segment.end)
            speaker_label = str(speaker_label)
            speaker_id = int(speaker_label.split("_")[-1]) if "_" in speaker_label else int(speaker_label)
            segments.append(SpeakerSegment(start=start, end=end, speaker_id=speaker_id))
    except AttributeError:
        # Not a pyannote Annotation — iterate directly
        for seg in annotation:
            if isinstance(seg, str):
                parts = seg.strip().split()
                kv = {p.split("=")[0]: p.split("=")[1] for p in parts if "=" in p}
                if kv:
                    start = float(kv["start"])
                    end = float(kv["end"])
                    speaker_label = kv["speaker"]
                else:
                    # Plain "start end speaker" whitespace-separated
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker_label = parts[2]
                speaker_id = int(speaker_label.split("_")[-1]) if "_" in speaker_label else int(speaker_label)
            elif isinstance(seg, tuple):
                # (start, end, speaker_label) tuple
                start = float(seg[0])
                end = float(seg[1])
                speaker_label = str(seg[2])
                speaker_id = int(speaker_label.split("_")[-1]) if "_" in speaker_label else int(speaker_label)
            else:
                start = float(seg.start)
                end = float(seg.end)
                speaker_label = str(seg.speaker) if hasattr(seg, "speaker") else str(seg.speaker_id)
                speaker_id = int(speaker_label.split("_")[-1]) if "_" in speaker_label else int(speaker_label)
            segments.append(SpeakerSegment(start=start, end=end, speaker_id=speaker_id))

    return segments


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    whisper_model: WhisperModel,
    audio_path: str,
    language: str | None = None,
) -> tuple[list[WordTimestamp], str]:
    """Transcribe audio and return word-level timestamps."""
    segments_iter, info = whisper_model.transcribe(
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


# ---------------------------------------------------------------------------
# Merging (mirrors the web app logic)
# ---------------------------------------------------------------------------

def merge_speakers_with_words(
    speaker_segments: list[SpeakerSegment],
    words: list[WordTimestamp],
) -> list[WordTimestamp]:
    """Assign a speaker_id to each word based on diarization segments."""
    if not words:
        return []
    if not speaker_segments:
        return [WordTimestamp(w.text, w.start, w.end, 0) for w in words]

    sorted_segs = sorted(speaker_segments, key=lambda s: s.start)

    result: list[WordTimestamp] = []
    for w in words:
        midpoint = (w.start + w.end) / 2

        # Find containing segment
        matched = None
        for seg in sorted_segs:
            if seg.start <= midpoint <= seg.end:
                matched = seg
                break

        if matched:
            result.append(WordTimestamp(w.text, w.start, w.end, matched.speaker_id))
            continue

        # Nearest segment fallback
        nearest = min(
            sorted_segs,
            key=lambda s: abs(midpoint - (s.start + s.end) / 2),
        )
        result.append(WordTimestamp(w.text, w.start, w.end, nearest.speaker_id))

    return result


def group_words_into_chunks(
    words: list[WordTimestamp],
) -> list[TranscriptChunk]:
    """Group consecutive words by speaker into transcript chunks."""
    if not words:
        return []

    chunks: list[TranscriptChunk] = []
    current_words: list[str] = [words[0].text]
    current_start = words[0].start
    current_end = words[0].end
    current_speaker = words[0].speaker_id

    for w in words[1:]:
        if w.speaker_id != current_speaker:
            chunks.append(TranscriptChunk(
                text=" ".join(current_words).strip(),
                start_time=current_start,
                end_time=current_end,
                speaker_id=current_speaker,
            ))
            current_words = [w.text]
            current_start = w.start
            current_end = w.end
            current_speaker = w.speaker_id
        else:
            current_words.append(w.text)
            current_end = w.end

    chunks.append(TranscriptChunk(
        text=" ".join(current_words).strip(),
        start_time=current_start,
        end_time=current_end,
        speaker_id=current_speaker,
    ))

    return chunks


def merge_consecutive_chunks(
    chunks: list[TranscriptChunk],
    gap_threshold: float = 1.0,
) -> list[TranscriptChunk]:
    """Merge adjacent chunks from the same speaker if gap is small."""
    if len(chunks) <= 1:
        return chunks

    merged = [TranscriptChunk(
        chunks[0].text, chunks[0].start_time,
        chunks[0].end_time, chunks[0].speaker_id
    )]

    for c in chunks[1:]:
        prev = merged[-1]
        gap = c.start_time - prev.end_time
        if c.speaker_id == prev.speaker_id and gap < gap_threshold:
            merged[-1] = TranscriptChunk(
                text=prev.text + " " + c.text,
                start_time=prev.start_time,
                end_time=max(prev.end_time, c.end_time),
                speaker_id=prev.speaker_id,
            )
        else:
            merged.append(TranscriptChunk(
                c.text, c.start_time, c.end_time, c.speaker_id
            ))

    return merged


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def ensure_wav_16k_mono(audio_path: str) -> str:
    """
    Convert audio to 16 kHz mono WAV if needed.
    Returns path to the (possibly converted) file.
    """
    data, sr = sf.read(audio_path)

    needs_conversion = False
    if sr != 16000:
        needs_conversion = True
    if data.ndim > 1:
        needs_conversion = True

    if not needs_conversion and audio_path.lower().endswith(".wav"):
        return audio_path

    # Convert
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, 16000)
    return tmp.name


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def output_text(chunks: list[TranscriptChunk]) -> str:
    lines: list[str] = []
    for c in chunks:
        ts = f"[{format_time(c.start_time)} - {format_time(c.end_time)}]"
        lines.append(f"Speaker {c.speaker_id} {ts}")
        lines.append(f"  {c.text}")
        lines.append("")
    return "\n".join(lines)


def output_vtt(chunks: list[TranscriptChunk]) -> str:
    lines = ["WEBVTT", ""]
    for c in chunks:
        start = format_vtt_time(c.start_time)
        end = format_vtt_time(c.end_time)
        lines.append(f"{start} --> {end}")
        lines.append(f"<v Speaker {c.speaker_id}>{c.text}")
        lines.append("")
    return "\n".join(lines)


def output_json(chunks: list[TranscriptChunk]) -> str:
    return json.dumps([asdict(c) for c in chunks], indent=2)


# ---------------------------------------------------------------------------
# YouTube support
# ---------------------------------------------------------------------------

_YOUTUBE_RE = re.compile(
    r"https?://(www\.|m\.|music\.)?(youtube\.com/(watch|shorts|live|embed)|youtu\.be/)",
)

_YOUTUBE_PLAYLIST_RE = re.compile(
    r"https?://(www\.|m\.|music\.)?youtube\.com/playlist\?list=",
)


def is_youtube_url(s: str) -> bool:
    """Return True if *s* looks like a YouTube URL (single video)."""
    return bool(_YOUTUBE_RE.match(s)) and not is_youtube_playlist_url(s)


def is_youtube_playlist_url(s: str) -> bool:
    """Return True if *s* looks like a YouTube playlist URL."""
    return bool(_YOUTUBE_PLAYLIST_RE.match(s))


def _sanitize_filename(name: str) -> str:
    """Remove characters that are unsafe in file names."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    if len(name) > 200:
        name = name[:200]
    return name or "transcript"


def download_youtube_audio(url: str) -> tuple[str, str]:
    """Download audio from a YouTube URL via yt-dlp.

    Returns ``(audio_path, video_title)``.
    """
    if not shutil.which("yt-dlp"):
        print(
            "Error: yt-dlp is required for YouTube URLs.\n"
            "  Install it with:  pip install yt-dlp",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fetch title
    proc = subprocess.run(
        ["yt-dlp", "--print", "title", url],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"Error fetching video info: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    title = proc.stdout.strip()

    # Download & extract audio as WAV
    tmp_dir = tempfile.mkdtemp(prefix="opensono_yt_")
    out_template = os.path.join(tmp_dir, "audio.%(ext)s")

    print(f"Downloading audio: {title}", file=sys.stderr)
    proc = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "wav", "-o", out_template, url],
    )
    if proc.returncode != 0:
        print("Error: yt-dlp failed to download audio", file=sys.stderr)
        sys.exit(1)

    audio_file = os.path.join(tmp_dir, "audio.wav")
    if not os.path.exists(audio_file):
        # Fallback: pick whatever file yt-dlp wrote
        files = os.listdir(tmp_dir)
        if files:
            audio_file = os.path.join(tmp_dir, files[0])
        else:
            print("Error: no audio file was downloaded", file=sys.stderr)
            sys.exit(1)

    return audio_file, title


def get_playlist_entries(url: str) -> tuple[list[tuple[str, str]], str]:
    """Fetch video URLs and titles from a YouTube playlist.

    Returns ``([(video_url, title), ...], playlist_title)``.
    """
    if not shutil.which("yt-dlp"):
        print(
            "Error: yt-dlp is required for YouTube URLs.\n"
            "  Install it with:  pip install yt-dlp",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get playlist title
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "playlist_title",
         "--playlist-items", "1", url],
        capture_output=True, text=True,
    )
    playlist_title = proc.stdout.strip() or "playlist"

    # Get all video URLs and titles
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "url", "--print", "title", url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"Error fetching playlist info: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    lines = proc.stdout.strip().splitlines()
    if len(lines) < 2:
        print("Error: playlist appears to be empty", file=sys.stderr)
        sys.exit(1)

    entries: list[tuple[str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        video_url = lines[i].strip()
        title = lines[i + 1].strip()
        entries.append((video_url, title))

    return entries, playlist_title


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SPEAKER_COLORS = [
    "\033[94m",  # blue
    "\033[92m",  # green
    "\033[95m",  # purple
    "\033[93m",  # orange/yellow
    "\033[91m",  # red/pink
]
RESET = "\033[0m"


def print_colored(chunks: list[TranscriptChunk]) -> None:
    for c in chunks:
        color = SPEAKER_COLORS[c.speaker_id % len(SPEAKER_COLORS)]
        ts = f"[{format_time(c.start_time)} - {format_time(c.end_time)}]"
        print(f"{color}Speaker {c.speaker_id} {ts}{RESET}")
        print(f"  {c.text}")
        print()


def _transcribe_file(
    audio_path: str,
    whisper_model: WhisperModel,
    diar_model,
    *,
    language: str | None,
    fmt: str,
    output_path: str | None,
) -> None:
    """Run the full transcribe + diarize + output pipeline on a single audio file."""
    print("Preparing audio...", file=sys.stderr)
    wav_path = ensure_wav_16k_mono(audio_path)
    tmp_created = wav_path != audio_path

    try:
        # Transcribe
        print("Transcribing...", file=sys.stderr)
        words, detected_lang = transcribe_audio(whisper_model, wav_path, language)
        print(f"  {len(words)} words transcribed", file=sys.stderr)

        # Diarize
        speaker_segments: list[SpeakerSegment] = []
        if diar_model is not None:
            print("Diarizing...", file=sys.stderr)
            speaker_segments = diarize_audio(diar_model, wav_path)
            print(f"  {len(speaker_segments)} speaker segments found", file=sys.stderr)

        # Merge
        words_with_speakers = merge_speakers_with_words(speaker_segments, words)
        chunks = group_words_into_chunks(words_with_speakers)
        chunks = merge_consecutive_chunks(chunks)

        # Output
        if fmt == "vtt":
            result = output_vtt(chunks)
        elif fmt == "json":
            result = output_json(chunks)
        else:
            result = output_text(chunks)

        if output_path:
            Path(output_path).write_text(result)
            print(f"Saved to {output_path}", file=sys.stderr)
        else:
            if fmt == "text" and sys.stdout.isatty():
                print_colored(chunks)
            else:
                print(result)

    finally:
        if tmp_created and os.path.exists(wav_path):
            os.unlink(wav_path)


def main():
    from opensono import __version__

    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio using Faster Whisper + NeMo Sortformer",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("audio", help="Path to audio file, YouTube URL, or YouTube playlist URL")
    parser.add_argument(
        "--model-size", default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device to run Whisper on (default: cuda)",
    )
    parser.add_argument(
        "--compute-type", default="float16",
        help="Compute type for Whisper (default: float16, use int8 for CPU)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Language code (e.g. en). Auto-detected if not set.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (or directory for playlists). Prints to stdout if not set.",
    )
    parser.add_argument(
        "--format", "-f", default="text",
        choices=["text", "vtt", "json"],
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip diarization (transcription only)",
    )

    args = parser.parse_args()

    ext = {"text": "txt", "vtt": "vtt", "json": "json"}[args.format]

    # Load models once up-front
    print(f"Loading Whisper model ({args.model_size})...", file=sys.stderr)
    whisper = WhisperModel(
        args.model_size, device=args.device, compute_type=args.compute_type
    )

    diar_model = None
    if not args.no_diarize:
        print("Loading diarization model...", file=sys.stderr)
        diar_model = load_diarization_model()

    # --- YouTube playlist ---
    if is_youtube_playlist_url(args.audio):
        entries, playlist_title = get_playlist_entries(args.audio)
        print(
            f"Playlist: {playlist_title} ({len(entries)} videos)",
            file=sys.stderr,
        )

        out_dir = args.output or _sanitize_filename(playlist_title)
        os.makedirs(out_dir, exist_ok=True)

        for idx, (video_url, title) in enumerate(entries, 1):
            print(
                f"\n{'='*60}\n[{idx}/{len(entries)}] {title}\n{'='*60}",
                file=sys.stderr,
            )
            yt_tmp_dir = None
            try:
                yt_audio_path, yt_title = download_youtube_audio(video_url)
                yt_tmp_dir = os.path.dirname(yt_audio_path)
                out_file = os.path.join(
                    out_dir, f"{_sanitize_filename(yt_title)}.{ext}"
                )
                _transcribe_file(
                    yt_audio_path,
                    whisper,
                    diar_model,
                    language=args.language,
                    fmt=args.format,
                    output_path=out_file,
                )
            except SystemExit:
                print(f"Skipping (download failed): {title}", file=sys.stderr)
                continue
            finally:
                if yt_tmp_dir and os.path.isdir(yt_tmp_dir):
                    shutil.rmtree(yt_tmp_dir, ignore_errors=True)

        print(f"\nAll transcripts saved to {out_dir}/", file=sys.stderr)
        return

    # --- Single YouTube video ---
    yt_tmp_dir = None
    if is_youtube_url(args.audio):
        yt_audio_path, yt_title = download_youtube_audio(args.audio)
        audio_path = yt_audio_path
        yt_tmp_dir = os.path.dirname(yt_audio_path)
        if not args.output:
            args.output = f"{_sanitize_filename(yt_title)}.{ext}"
    else:
        audio_path = str(Path(args.audio).resolve())
        if not Path(audio_path).exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

    try:
        _transcribe_file(
            audio_path,
            whisper,
            diar_model,
            language=args.language,
            fmt=args.format,
            output_path=args.output,
        )
    finally:
        if yt_tmp_dir and os.path.isdir(yt_tmp_dir):
            shutil.rmtree(yt_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
