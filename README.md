# OpenSono

**Open-source voice AI.**

Transcribe audio files with word-level timestamps and automatic speaker identification using [NVIDIA Parakeet CTC](https://huggingface.co/nvidia/parakeet-ctc-0.6b) (default) or [Faster Whisper](https://github.com/SYSTRAN/faster-whisper), combined with [NVIDIA NeMo Sortformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html) diarization.

> This is the Python CLI companion to [OpenSono WebApp](https://opensono.vercel.app) — the free, browser-based transcription tool.

## Features

- **Pluggable ASR backends** — Parakeet CTC 0.6B (default, English) or Faster Whisper (multilingual)
- **Speaker diarization** — Automatically identifies up to 4 speakers using NVIDIA Sortformer
- **Word-level timestamps** — Precise timing for every word
- **Multiple output formats** — Plain text, VTT subtitles, or JSON
- **Auto language detection** — Supported by the Whisper backend (99+ languages)
- **Colored terminal output** — Speaker-coded output for easy reading
- **YouTube support** — Transcribe directly from a YouTube URL or playlist (powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp))

## Installation

### GPU (default)

```bash
pip install opensono
# or, explicit:
pip install "opensono[gpu]"
```

On Linux this installs PyTorch's CUDA-enabled wheel from PyPI.

### CPU-only

With [uv](https://github.com/astral-sh/uv) (recommended — no extra flags needed):

```bash
uv pip install "opensono[cpu]"
```

The PyTorch CPU wheel index is baked into `pyproject.toml` via `[tool.uv.sources]` and resolved automatically.

With plain pip, you have to pass the index URL yourself:

```bash
pip install "opensono[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

### From GitHub (latest development version)

```bash
pip install git+https://github.com/penkow/opensono.git
```

> **Note:** The NeMo toolkit has additional system dependencies. See the [NeMo installation guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/getting-started.html) for details.

Requires Python 3.10+. A CUDA-capable GPU is recommended but not required — the CLI auto-detects CUDA and falls back to CPU.

### From source

```bash
git clone https://github.com/penkow/opensono
cd opensono
pip install .
```

## Usage

After installing, the `opensono` command is available anywhere in your terminal.

### Basic transcription with speaker diarization

```bash
opensono meeting.wav
```

### Transcribe a YouTube video

```bash
opensono "https://www.youtube.com/watch?v=VIDEO_ID"
```

The transcript is saved to a file named after the video title (e.g. `My Video Title.txt`).

### Transcribe a YouTube playlist

```bash
opensono "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

Each video in the playlist is downloaded, transcribed, and saved to a directory named after the playlist. Use `-o` to specify a custom output directory:

```bash
opensono "https://www.youtube.com/playlist?list=PLAYLIST_ID" -o my_transcripts
```

### Transcription only (no diarization)

```bash
opensono interview.mp3 --no-diarize
```

### Export as VTT subtitles

```bash
opensono podcast.wav -f vtt -o subtitles.vtt
```

### Export as JSON

```bash
opensono recording.wav -f json -o transcript.json
```

### Specify language (skip auto-detection)

```bash
opensono audio.wav --language en
```

### Switch backend

The default backend is `parakeet` (NVIDIA Parakeet CTC 0.6B, English-only). To use the multilingual Whisper backend:

```bash
opensono audio.wav --backend faster-whisper
```

### Use a different model

```bash
# A smaller/faster Whisper model
opensono audio.wav --backend faster-whisper --model base

# A different Parakeet checkpoint
opensono audio.wav --model nvidia/parakeet-rnnt-1.1b
```

### CPU-only

```bash
opensono audio.wav --backend faster-whisper --device cpu --compute-type int8
```

### Check version

```bash
opensono --version
```

You can also run it as a Python module:

```bash
python -m opensono audio.wav
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `parakeet` | Transcription backend (`parakeet`, `faster-whisper`) |
| `--model` | backend default | Model name (`nvidia/parakeet-ctc-0.6b` for parakeet, `large-v3` for faster-whisper) |
| `--device` | `auto` | Compute device (`auto`, `cuda`, `cpu`) — `auto` picks `cuda` if available, else `cpu` |
| `--compute-type` | `float16` | Precision for faster-whisper (`float16`, `int8`, `float32`); ignored for parakeet |
| `--language` | auto-detect | Language code (e.g. `en`, `fr`, `de`); faster-whisper only |
| `--format`, `-f` | `text` | Output format (`text`, `vtt`, `json`) |
| `--output`, `-o` | stdout | Output file path (or directory for playlists) |
| `--no-diarize` | off | Skip speaker diarization |
| `--chunk-len` | `300` | Parakeet chunk length in seconds for long audio (lower if you hit OOM); ignored by faster-whisper |

## Output formats

### Text (default)

```
Speaker 0 [0:00 - 0:03]
  Hello, welcome to the meeting.

Speaker 1 [0:03 - 0:07]
  Thanks for having me. Let's get started.
```

### VTT

```
WEBVTT

00:00:00.000 --> 00:00:03.500
<v Speaker 0>Hello, welcome to the meeting.

00:00:03.500 --> 00:00:07.200
<v Speaker 1>Thanks for having me. Let's get started.
```

### JSON

```json
[
  {
    "text": "Hello, welcome to the meeting.",
    "start_time": 0.0,
    "end_time": 3.5,
    "speaker_id": 0
  }
]
```

## How it works

1. **Audio preprocessing** — Converts input to 16 kHz mono WAV
2. **Transcription** — Selected backend produces word-level timestamps
3. **Diarization** — NeMo Sortformer identifies speaker segments
4. **Merging** — Each word is assigned to a speaker based on temporal overlap
5. **Grouping** — Consecutive words from the same speaker are combined into chunks

## Models

| Component | Model | Size |
|-----------|-------|------|
| Transcription (default) | [NVIDIA Parakeet CTC 0.6B](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | ~600 MB |
| Transcription (optional) | [Faster Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | ~3 GB |
| Diarization | [NVIDIA Sortformer 4spk v2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_streaming_sortformer_4spk-v2.1) | ~100 MB |

Models are downloaded automatically on first run and cached locally.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- ~2 GB VRAM for Parakeet CTC 0.6B; ~4 GB VRAM for Whisper large-v3

## Browser version

Don't want to install anything? Use [OpenSono WebApp](https://opensono.vercel.app) — the same transcription engine running entirely in your browser. No uploads, no sign-up, completely private.

## Roadmap

- [x] Word-level timestamps
- [x] Speaker diarization with NVIDIA Sortformer
- [x] Multiple output formats (text, VTT, JSON)
- [x] Auto language detection
- [ ] Improve the Sortformer config for non-streaming mode for better accuracy
- [ ] Support more than 4 speakers
- [ ] SRT subtitle format
- [ ] Streaming / real-time transcription
- [ ] REST API / server mode
- [ ] Python library API (importable, not just CLI)
- [ ] Speaker name assignment
- [ ] Diarization without GPU (CPU-compatible alternative)

## License

MIT — see [LICENSE](LICENSE) for details.
