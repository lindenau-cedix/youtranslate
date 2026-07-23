# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`youtranslate` is a single-file CLI tool that downloads an English YouTube
video, transcribes it, translates the speech to German (or another language),
and burns the translated subtitles into the video. There is no package, no
test suite, no build step — just `youtranslate.py` plus a dependency manifest.

## Pipeline architecture

`youtranslate.py:main()` runs four sequential stages, each in its own
top-level function. Stages communicate via `Segment` (a frozen dataclass
with `start: float`, `end: float`, `text: str`) so the two whisper engines
and the two translators stay swappable.

| Stage | Function | Default backend | Swap with |
|---|---|---|---|
| 1. Download | `download()` | `yt-dlp` | (fixed) |
| 2. Transcribe | `transcribe_faster_whisper()` / `transcribe_openai_whisper()` | **`openai-whisper`** (local) | `--engine` |
| 3. Translate | `translate_segments_deep()` / `translate_segments_openai()` | **`openai`** (Chat Completions) | `--translator` |
| 4. Burn / attach | `burn_subtitles()` (libass) or the soft-sub branch in `main()` | ffmpeg + libass, hardcoded | `--soft-subs` |

The two backends in stage 3 are deliberately asymmetric: `deep` is Google
Translate per-segment via `deep_translator.GoogleTranslator` (free, no key,
slower per call). `openai` uses `gpt-4o-mini` (override via
`OPENAI_TRANSLATE_MODEL`). Both translate per-segment to preserve
Whisper's exact timing — do not "improve" this by translating the whole
transcript and re-aligning; it's a documented trade-off in the README.

## Conventions worth knowing

- **External CLIs are validated with `ensure_tool()`.** It calls
  `shutil.which()` and raises `SystemExit` with a friendly message. Any
  new stage that shells out should follow this pattern, not bare
  `subprocess.run`.
- **Time formatting lives in `fmt_srt_time()` (seconds → `HH:MM:SS,mmm`)**.
  Subtitle writing goes through `write_srt()`, which also calls
  `wrap_text()` (assumes width 80 by default). Text wrapping matters
  because ffmpeg/libass will not wrap long lines automatically.
- **`download()` finds the output file in two ways**: first it scans
  yt-dlp's stdout for a `[Destination] / [Merged]` line; failing that, it
  picks the most recently modified `.mp4`/`.mkv`/`.webm` in the output
  directory. Keep both paths when changing this.
- **Per-segment translation retries 3× with exponential-ish backoff**
  inside `translate_segments_deep()`. The catch is intentionally wide
  (`except Exception`) because deep-translator raises a variety of types
  on Google rate-limits.
- **The OpenAI path fails fast.** `main()` checks `OPENAI_API_KEY` up
  front and exits with a setup hint before any work is done. If you add
  another secret-keyed backend, follow this pattern — the alternative
  (failing on first call) is confusing.
- **libass `force_style` in `burn_subtitles()`** uses ASS color hex
  (`&H00BBGGRR&` with full alpha being `&H00` — note the inverted RGB).
  Auto font size is `height / 28` clamped to `[16, 48]`. H.264 CRF 22
  is hard-coded; if the user wants the source video bit-exactly, they
  must use `--soft-subs` (mov_text stream), not a flag here.

## Running / developing

This is a script, not a package — `python3 youtranslate.py` works from
the repo root with the deps installed. There are no tests; the smoke
test is the help output.

```bash
# Install deps.
pip install -r requirements.txt

# System tools (not pip): yt-dlp is in requirements.txt, but ffmpeg and
# ffprobe must be on PATH with libass enabled.

# Syntax check / smoke.
python3 -c "import ast; ast.parse(open('youtranslate.py').read())"
python3 youtranslate.py --help
```

Deps for the optional backends (`openai-whisper`, `openai`,
`faster-whisper`, `deep-translator`) are all pinned only by name in
`requirements.txt`. Backends not used by current CLI flags may go
unimported; but they are still installed by `pip install -r
requirements.txt` so the user can swap backends without re-installing.
If you prune unused deps, do it via comments in `requirements.txt`,
not by removing lines — the readme still references both code paths.
