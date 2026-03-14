# OpenSono

**Open-source audio transcription with speaker diarization.**

Transcribe audio files with word-level timestamps and automatic speaker identification using [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) and [NVIDIA NeMo Sortformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html).

> This is the Python CLI companion to [OpenSono WebApp](https://opensono.vercel.app) — the free, browser-based transcription tool.

## Features

- **Accurate transcription** — Powered by Whisper large-v3
- **Speaker diarization** — Automatically identifies up to 4 speakers using NVIDIA Sortformer
- **Word-level timestamps** — Precise timing for every word
- **Multiple output formats** — Plain text, VTT subtitles, or JSON
- **Auto language detection** — Supports 99+ languages
- **Colored terminal output** — Speaker-coded output for easy reading

## Installation

```bash
pip install opensono
```

> **Note:** The NeMo toolkit has additional system dependencies. See the [NeMo installation guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/getting-started.html) for details.

Requires Python 3.10+ and a CUDA-capable GPU (recommended).

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

### Use a smaller/faster model

```bash
opensono audio.wav --model-size base
```

### CPU-only

```bash
opensono audio.wav --device cpu --compute-type int8
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
| `--model-size` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--device` | `cuda` | Compute device (`cuda` or `cpu`) |
| `--compute-type` | `float16` | Precision (`float16`, `int8`, `float32`) |
| `--language` | auto-detect | Language code (e.g. `en`, `fr`, `de`) |
| `--format`, `-f` | `text` | Output format (`text`, `vtt`, `json`) |
| `--output`, `-o` | stdout | Output file path |
| `--no-diarize` | off | Skip speaker diarization |

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
2. **Transcription** — Faster Whisper produces word-level timestamps
3. **Diarization** — NeMo Sortformer identifies speaker segments
4. **Merging** — Each word is assigned to a speaker based on temporal overlap
5. **Grouping** — Consecutive words from the same speaker are combined into chunks

## Models

| Component | Model | Size |
|-----------|-------|------|
| Transcription | [Faster Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | ~3 GB |
| Diarization | [NVIDIA Sortformer 4spk v2.1](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_streaming_sortformer_4spk-v2.1) | ~100 MB |

Models are downloaded automatically on first run and cached locally.

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- ~4 GB VRAM for GPU inference with large-v3

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
