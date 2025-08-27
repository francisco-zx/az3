Audio Transcription + Speaker Diarization (Whisper + pyannote)
=============================================================

Transcribe audio and label speakers using OpenAI Whisper and pyannote.audio. Includes persistent speaker profiles, optional audio enhancement, configurable accuracy knobs, and optional AI-generated summaries.

Features
--------
- Whisper transcription per diarized segment
- pyannote 3.1 speaker diarization (GPU-aware)
- Persistent speaker profiles with cross-file matching (ECAPA embeddings)
- JSON/SRT/TXT outputs
- Optional audio enhancement (pre-emphasis, band-pass, denoise)
- Language auto-detect + lock for stability (or force with `-l`)
- AI summaries (OpenAI) or built-in heuristic summary
- Structured logging with `--log-level`

Requirements
------------
- Python 3.9+
- FFmpeg (for `librosa` reading many formats)
- Python packages (see `requirements.txt`) plus:
  - `scikit-learn` (already in requirements)
  - `scipy` (for enhancement)
  - `speechbrain` (preferred for speaker embeddings)
  - `openai` (optional, for AI summaries)

Install system deps (macOS):
```
brew install ffmpeg
```

Create venv and install:
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install scipy speechbrain openai
```

Authentication
--------------
pyannote diarization model is gated; set a Hugging Face token and accept model terms.
```
# Create a Read token at https://huggingface.co/settings/tokens and accept
# https://huggingface.co/pyannote/speaker-diarization-3.1
export HUGGINGFACE_HUB_TOKEN=hf_...
# or
export HF_TOKEN=hf_...
```
Optional (AI summaries):
```
export OPENAI_API_KEY=sk-...
```

Quick start
-----------
```
python main.py /path/to/audio.wav -f json --log-level INFO
```
With enhancements and better decoding:
```
python main.py /path/to/audio.mp3 \
  --enhance light --boost-accuracy --segment-padding 0.75 \
  --save-signatures --log-level INFO
```
Generate an AI summary:
```
python main.py /path/to/audio.wav --summarize --ai-summary \
  --summary-file output/summary.txt --ai-model gpt-4o-mini --ai-temperature 0.3
```

CLI reference
-------------
- `audio_file` (positional): Path to input audio
- `-o, --output`: Output path (without extension)
- `-f, --format`: `json` (default) | `srt` | `txt`
- `-w, --whisper-model`: `tiny|base|small|medium|large` (default: `base`)
- `-l, --language`: Force language (e.g., `en`, `es`). When omitted, first minute is used to auto-detect and lock
- `--min-speakers`: Min expected speakers (default: 1)
- `--max-speakers`: Max expected speakers (default: 10)
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL` (default: `INFO`)
- `--save-signatures`: Export average speaker embeddings to JSON
- `--signatures-file`: Path for signatures JSON (default: `speaker_profiles/signatures.json`)
- `--boost-accuracy`: Whisper decoding tweaks (temperature=0, no previous-text conditioning)
- `--segment-padding`: Seconds of context around segments (default: 0.5)
- `--enhance`: `off|light|medium` (default: `off`). `light`=pre-emphasis+band-pass; `medium` adds denoise
- `--similarity-threshold`: Cosine similarity threshold for cross-file speaker matching (default: 0.72)
- `--summarize`: Produce conversation summary
- `--summary-file`: Summary output (default: `output/summary.txt`)
- `--ai-summary`: Use OpenAI to generate higher-quality summary
- `--ai-model`: OpenAI chat model (default: `gpt-4o-mini`)
- `--ai-max-tokens`: Max tokens for AI summary (default: 900)
- `--ai-temperature`: Decoding temperature for AI summary (default: 0.3)

Outputs
-------
- JSON: `metadata` + `segments` array. Each segment: `{start_time, end_time, speaker, text}`
- SRT: Numbered blocks with `[Speaker]` prefix
- TXT: Timestamped lines with `speaker: text`
- Summary: `output/summary.txt` (either heuristic or AI-based)

Speaker profiles
----------------
- Stored under `speaker_profiles/`:
  - `speaker_profiles.pkl` (binary profiles)
  - `profiles_metadata.json` (human-readable summary)
  - `signatures.json` (L2-normalized average embedding per speaker)
- Matching behavior:
  - Multiple segment embeddings per temporary speaker are normalized and averaged, then matched to existing profiles via cosine similarity
  - Tune with `--similarity-threshold` (0.65–0.75 typical). Lower is more permissive
- Programmatic corrections (examples inside `main.py`):
  - Assign name: `SpeakerProfileManager.assign_speaker_name("Speaker_001", "Alice")`
  - Merge: `SpeakerProfileManager.merge_speakers("Speaker_001", "Speaker_003")`

Modules overview
----------------
- `AudioProcessor`
  - Load audio at 16 kHz; optional enhancement (pre-emphasis, band-pass, denoise)
- `SpeakerDiarizer`
  - Loads `pyannote/speaker-diarization-3.1` (uses GPU if available)
- `WhisperTranscriber`
  - Loads Whisper; detects and locks language if not provided; pads segments for context; accuracy tweak options
- `SpeakerProfileManager`
  - Loads/saves persistent profiles; extracts ECAPA embeddings; averages and matches across files
  - Exports JSON signatures; supports naming/merging
- `AudioTranscriberDiarizer`
  - Orchestrates: load → (optional enhance) → diarize → per-segment transcribe → map speakers → save results → (optional) summary

Accuracy & performance tips
---------------------------
- Prefer larger Whisper models (`-w medium|large`) for accuracy (at cost of speed)
- Provide language with `-l` to skip detection (stabilizes recognition)
- Use `--segment-padding 0.75` to reduce word truncation
- Start with `--enhance light` for speed; use `--enhance medium` only for very noisy clips
- Reduce `--max-speakers` if you know the count
- Use GPU when possible (both Whisper and pyannote benefit)

Troubleshooting
---------------
- Diarization auth error: set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) and accept the model card terms
- Import errors in IDE: ensure venv is active and `pip install` ran successfully
- Slow/stuck during enhancement: run with `--enhance off` or `--enhance light`
- Speaker not recognized across files: lower `--similarity-threshold` (e.g., 0.68), ensure profiles persist between runs, and that audio isn’t too short/noisy
- AI summary error: set `OPENAI_API_KEY` and install `openai`

Programmatic usage (minimal)
----------------------------
```
from main import AudioTranscriberDiarizer
transcriber = AudioTranscriberDiarizer(whisper_model="base", language=None)
segments = transcriber.process_audio("audio.wav", min_speakers=1, max_speakers=5)
transcriber.save_results(segments, "output/transcript.json", "json")
summary = transcriber.summarize_conversation(segments)
open("output/summary.txt", "w").write(summary)
```

License
-------
This project uses third-party models and libraries subject to their respective licenses (OpenAI Whisper, pyannote.audio, SpeechBrain, etc.). Review their terms before use.

