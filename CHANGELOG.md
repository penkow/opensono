# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4] - 2026-05-16

### Added

- Parakeet long-audio chunking — audio longer than `--chunk-len` seconds
  (default 300) is split into chunks, transcribed one at a time, and stitched
  back together with global timestamps. Keeps peak VRAM bounded for hour-long files.
- `--chunk-len` CLI flag to tune chunk size (lower it for tighter GPUs)

## [0.2.3] - 2026-05-16

### Fixed

- Release GPU memory between transcribe/diarize stages, between playlist
  iterations, and after `main()` finishes — fixes CUDA OOM on tight-VRAM setups
  and long playlist runs

## [0.2.2] - 2026-05-16

### Changed

- `yt-dlp` is now a hard dependency — YouTube URLs and playlists work out of the box
- Removed the runtime "yt-dlp is required" error and its install hint

## [0.2.1] - 2026-05-16

### Added

- `[cpu]` and `[gpu]` install extras (`pip install opensono[cpu]` / `[gpu]`)
- uv-aware `[tool.uv.sources]` config so `uv pip install opensono[cpu]` resolves
  torch from the PyTorch CPU wheel index automatically — no `--extra-index-url` flag
- `--device auto` (now the default): picks `cuda` if available, otherwise falls back to `cpu`

### Changed

- Diarization model (`load_diarization_model`) now respects the selected device
  instead of always loading on CUDA

## [0.2.0] - 2026-05-16

### Added

- Pluggable transcription backend abstraction (`opensono.transcribers`) with a `Transcriber` ABC
- NVIDIA Parakeet CTC 0.6B backend (`nvidia/parakeet-ctc-0.6b`) — now the default
- `--backend {parakeet,faster-whisper}` CLI flag

### Changed

- Default transcription backend is now Parakeet CTC 0.6B (was Faster Whisper large-v3)
- Renamed `--model-size` to `--model` (old flag kept as an alias for backward compatibility)
- `core.py` decoupled from `faster_whisper.WhisperModel`; transcription goes through the `Transcriber` interface

## [0.1.6] - 2026-04-13

### Added

- YouTube playlist support — pass a playlist URL to transcribe all videos
- Each video is downloaded, transcribed, and saved to a directory named after the playlist
- Models are loaded once and reused across all playlist videos

### Changed

- Refactored transcription pipeline into reusable `_transcribe_file()` function

## [0.1.5] - 2026-04-13

### Added

- YouTube URL support — pass a YouTube URL instead of a file path to transcribe directly
- Audio is downloaded via `yt-dlp` and transcript is saved using the video title as filename
- `pip install git+...` install method documented in README

## [0.1.4] - 2026-03-14

### Changed

- Bump version to `0.1.4` to test pipelines

## [0.1.3] - 2026-03-14

### Fixed

- Bump `pyproject.toml` version to `0.1.3` to match release tag

## [0.1.2] - 2026-03-14

### Added

- Roadmap section in `README.md`

### Changed

- Updated `.gitignore`

## [0.1.1] - 2026-03-14

### Fixed

- Handle `pyannote` `Annotation` output from NeMo diarizer

### Changed

- Update homepage URL to `opensono.vercel.app`

## [0.1.0] - 2026-03-13

### Added

- Initial release with audio transcription and speaker diarization
- Support for `faster-whisper` transcription backend
- NeMo-based speaker diarization
- PyPI publish workflow