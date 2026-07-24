# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Letzter Durchlauf

Echtes **Dubbing** ergänzt (`--dub`): Statt nur Untertitel einzublenden, werden
die Originalstimmen durch generierte, übersetzte Sprache ersetzt, während
Musik/Umgebung/Geräusche erhalten bleiben. Neue Stufe 5 in `youtranslate.py`:
`extract_audio` → `separate_vocals` (Demucs `--two-stems=vocals`, `no_vocals`
= Hintergrund) → TTS pro Segment (`synthesize_elevenlabs`/`synthesize_openai`
über stdlib-`urllib`, Standard **ElevenLabs**) → `fit_clip` (atempo, nur
Beschleunigung bis `--dub-max-tempo`, Kappe 2.0) → `assemble_and_mix`
(numpy/soundfile: Clips auf stiller Timeline platzieren, über Hintergrund
mischen, Peak-Normalisierung) → Mux. Optionales **Voice-Cloning** der
Originalstimme via ElevenLabs IVC (`--clone-voice`, löscht die temporäre Stimme
wieder außer bei `--keep-clone`).

Untertitel sind jetzt ein eigenes Flag: `--subtitles {burn,soft,none}`
(Default `burn`, rückwärtskompatibel; `--soft-subs` bleibt als veralteter Alias).
Dub und Untertitel sind unabhängig kombinierbar; mindestens eines muss aktiv
sein. `burn_subtitles`/neue `attach_or_replace` akzeptieren einen optionalen
`audio_path`, um die gedubbte Tonspur einzumuxen. Fail-fast für
`ELEVENLABS_API_KEY`/`OPENAI_API_KEY` analog zum bestehenden Muster.

Verifiziert ohne Netz/echte Medien: Syntax + `--help`; Offline-Unit-Tests für
`plan_tempo`, `iso1`, Voice-ID-Heuristik, `language_code`-Gating, Dateinamen,
Multipart und CLI-Validierung; ffmpeg-/demucs-Kommandozeilen über Testdoubles
(alle sechs Render-Pfade, extract/separate, Device-Flag, Stem-Pfade);
`assemble_and_mix` mit echtem numpy/soundfile (Platzierung, Mono→Stereo,
Overflow-Clamping, Peak-Normalisierung); ElevenLabs-HTTP-Vertrag (TTS-URL,
Header, Body, Cloning-Multipart, Delete) gegen einen lokalen Mock-Server.

Vorheriger Durchlauf: HTTP-403-Fehler beim YouTube-Download adressiert
(`--cookies-from-browser`, reicht die vollständige Browser-Spezifikation an
yt-dlp weiter; 403-Hinweis bei unauthentifiziertem Download).

## What this is

`youtranslate` is a single-file CLI tool that downloads a YouTube video in
any language, transcribes it, translates the speech to another language, and
then either burns the translated subtitles into the video **or dubs it**
(replaces the original voices with generated, translated speech while keeping
the background audio). There is no package, no test suite, no build step —
just `youtranslate.py` plus a dependency manifest. The "smoke test" is the
`--help` output; pure logic is checked with ad-hoc offline scripts and CLI
testdoubles (see the last "Letzter Durchlauf" for the pattern).

## Pipeline architecture

`youtranslate.py:main()` runs a sequence of stages, each in its own
top-level function. Stages communicate via `Segment` (a frozen dataclass
with `start: float`, `end: float`, `text: str`) so the two whisper engines,
the two translators, and the two TTS backends stay swappable. Steps 1–3 always
run; step 4 is subtitles and/or dubbing (at least one must be enabled) — the
`[n/N]` step counter in `main()` is computed dynamically because `--dub` adds a
stage.

| Stage | Function | Default backend | Swap with |
|---|---|---|---|
| 1. Download | `download()` | `yt-dlp` | (fixed) |
| 2. Transcribe | `transcribe_faster_whisper()` / `transcribe_openai_whisper()` | **`openai-whisper`** (local) | `--engine` |
| 3. Translate | `translate_segments_deep()` / `translate_segments_openai()` | **`openai`** (Chat Completions) | `--translator` |
| 4a. Subtitles | `burn_subtitles()` (libass + ffmpeg) or `attach_or_replace()` (soft mov_text) | ffmpeg + libass; `h264_nvenc` if available, else `libx264` | `--subtitles {burn,soft,none}`, `--burn-encoder` |
| 4b. Dub | `dub_audio()` → `separate_vocals` + `synthesize_segment` + `fit_clip` + `assemble_and_mix` | Demucs + **ElevenLabs** TTS | `--dub`, `--tts {elevenlabs,openai}`, `--clone-voice` |

Subtitles and dubbing are independent and compose: the renderer picks
`burn_subtitles()` when `--subtitles burn` (folding in the dubbed audio via its
`audio_path` param during the same re-encode) and `attach_or_replace()`
otherwise (video stream-copied; replaces audio and/or attaches soft subs). The
output name is built by `compose_output_name()`:
`<base>.<src>.<tgt>[.dubbed][.subtitled].mp4`.

`transcribe()` returns `(segments, detected_lang)` — both backends expose
Whisper's detected source code, and `main()` uses it for filenames and
status printout when the user didn't pin `--source-lang`.

The two backends in stage 3 are deliberately asymmetric: `deep` is Google
Translate per-segment via `deep_translator.GoogleTranslator` (free, no key,
slower per call). `openai` uses `gpt-4o-mini` (override via
`OPENAI_TRANSLATE_MODEL`). Both translate per-segment to preserve
Whisper's exact timing — do not "improve" this by translating the whole
transcript and re-aligning; it's a documented trade-off in the README.

Stage 4a has a GPU path: `ffmpeg_has_encoder(name)` probes the system ffmpeg
for an encoder (e.g. `h264_nvenc`) by parsing `ffmpeg -encoders` output.
`resolve_encoder()` (shared, so the NVENC/libx264 decision lives in one place)
maps `--burn-encoder {auto,nvenc,libx264}` to a bool, then `burn_subtitles()`
chooses `h264_nvenc` with `-rc vbr -cq 22 -b:v 0 -preset p4` (5–15× faster than
libx264 with no perceptual quality loss), or falls back to `libx264 -crf 22`.
NVENC requires an ffmpeg built with `--enable-nvenc`; the README has the full
Debian/Ubuntu build recipe. `--subtitles soft` (or the deprecated `--soft-subs`
alias) skips the re-encode entirely — `attach_or_replace()` attaches a
`mov_text` stream and stream-copies the video.

### Dubbing stage (`dub_audio`, step 4b)

- **Separation is the trick for "keep the background".** `separate_vocals()`
  runs Demucs `--two-stems=vocals`; the `no_vocals.wav` stem (music + ambience +
  noise) is kept and the `vocals.wav` stem (original speech) is discarded — or,
  with `--clone-voice`, used as the cloning reference. Stems land at
  `<out>/<demucs-model>/<input-stem>/{vocals,no_vocals}.wav`; keep that path
  logic if you change the model flag.
- **TTS backends mirror the translator pattern.** `synthesize_segment()`
  dispatches to `synthesize_elevenlabs()` / `synthesize_openai()`. Both talk raw
  HTTP over **stdlib `urllib`** (no SDK — avoids ElevenLabs/OpenAI SDK version
  churn); `_http_bytes()` centralizes the POST + error surfacing. ElevenLabs
  `language_code` is only sent for models where `_model_accepts_language_code()`
  is true (turbo/flash/v3) — `eleven_multilingual_v2` auto-detects and rejects
  it. Voice names are resolved to IDs via `elevenlabs_resolve_voice()`
  (`_looks_like_voice_id()` skips the API call for 20-char IDs).
- **Per-segment timing is preserved, same as translation.** `fit_clip()` only
  ever *speeds up* a clip (`plan_tempo()` returns 1.0 when it already fits) via
  `atempo`, capped by `--dub-max-tempo` (validated ≤ 2.0 because a single
  `atempo` maxes at 2.0). Do not "improve" this by re-timing the whole track.
- **Mixing is numpy, not an ffmpeg `amix` of hundreds of inputs.**
  `assemble_and_mix()` (lazy `import numpy, soundfile`) lays each fitted clip
  onto a silent stereo timeline at `round(start*DUB_SR)`, sums overlaps,
  mixes over the background at `--dub-bg-gain`/`--dub-voice-gain`, and
  peak-normalizes to avoid clipping. Everything is pinned to `DUB_SR`=44100 /
  stereo (Demucs' native rate) so arrays line up without resampling in Python.
- **Voice cloning is opt-in and self-cleaning.** `--clone-voice` uploads a
  ≤120 s sample of the separated vocals via `elevenlabs_clone_voice()`
  (hand-rolled multipart in `_multipart()`), then `dub_audio()` deletes the
  temporary voice in a `finally` unless `--keep-clone`. Requires a paid plan;
  the README carries the consent warning.
- **Heavy imports stay lazy.** numpy/soundfile/whisper/openai/etc. are imported
  inside their functions so `--help`, argument validation, and offline logic
  tests run without them installed.

## Languages

Source and target languages are independent.

- `--source-lang` (default `auto`): ISO-639-1 code or `auto` to let Whisper
  detect from audio. With `auto`, the `language=` kwarg is dropped from
  Whisper's `transcribe()` call so detection runs.
- `--target-lang` (default `de`): any code `deep_translator` or `babel` can
  handle, including regional tags like `pt-BR`. It also drives TTS in the dub
  stage; `iso1()` reduces it to a bare ISO-639-1 code (`pt-BR` → `pt`) for the
  ElevenLabs `language_code` param.
- `human_lang_name(code)` resolves a code to its English display name via
  `babel.Locale.parse(...).get_display_name("en")`. Used for the OpenAI
  translator prompt and the final printout. Falls back to the raw code if
  `babel` is missing or the code is unparseable — never raises.
- Output filenames follow `<title>.<src>.<tgt>.<ext>`: `<src>` is
  `--source-lang` if pinned, otherwise the language Whisper detected.

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
  directory. Keep both paths when changing this. `--cookies-from-browser`
  forwards yt-dlp's complete browser specification (for example `chrome` or
  `chrome:Profile 2`) without copying cookies into the project. When an
  unauthenticated download reports 403/Forbidden, `download()` suggests the
  Chrome option; it does not retry automatically or silently use credentials.
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
  Auto font size is `height / 100` clamped to `[10, 24]` for compact default
  subtitles. Pass `--font-size` to override it explicitly. The video
  encoder is no longer hardcoded — `--burn-encoder {auto,nvenc,libx264}`
  picks between NVENC (quality knob: `-cq 22`) and libx264 (CRF 22). If
  the user wants the source video bit-exactly, they must use `--subtitles soft`
  (mov_text stream) or `--dub` without burning, not a flag here.

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

### Verifying changes (no test suite, deps/network usually absent)

Heavy deps (whisper/openai/demucs/numpy/soundfile) and the network are normally
unavailable in the dev sandbox, and `import youtranslate` succeeds without them
because every heavy import is lazy. Verify accordingly — this is the pattern the
"Letzter Durchlauf" refers to:

- **Pure logic, offline.** Import the module and call functions directly:
  `plan_tempo`, `iso1`, `_looks_like_voice_id`, `_model_accepts_language_code`,
  `compose_output_name`, `_multipart`, and the `main()` validation branches
  (missing `*_API_KEY`, `--subtitles none` without `--dub`, `--dub-max-tempo`
  out of range) all run with nothing installed. Catch `SystemExit` to assert the
  validation messages.
- **ffmpeg / demucs command assembly.** Put fake `ffmpeg`/`ffprobe`/`demucs`
  scripts on `PATH` that append `"$@"` to a log and `touch` their final arg. Make
  the fake `ffprobe` answer `stream=width,height` (e.g. `1920x1080`) and
  `format=duration`, and `ffmpeg -encoders` print or omit `h264_nvenc` to force
  the NVENC vs libx264 branch. Then drive `burn_subtitles()` /
  `attach_or_replace()` / `separate_vocals()` / `fit_clip()` and assert the
  captured argv — cover all six render paths (burn/soft/none × ±dub) plus the
  demucs stem-path resolution.
- **ElevenLabs HTTP contract.** Reassign `youtranslate.ELEVENLABS_API` to a local
  `http.server` mock, then call `synthesize_elevenlabs`, `elevenlabs_resolve_voice`
  (by name → GET `/voices`, by ID → no call), `elevenlabs_clone_voice` and
  `elevenlabs_delete_voice`; assert URL, `xi-api-key`/`Accept` headers, JSON body
  (incl. `language_code` gating) and the multipart upload.
- **numpy mixing (needs real deps).** `/opt/venv` is read-only, so create a
  throwaway venv (`python3 -m venv <scratch>/venv && .../pip install numpy
  soundfile`), write tiny wavs, and assert `assemble_and_mix()` placement,
  mono→stereo upmix, overflow clamping to the background length, and peak
  normalization to 1.0.

Deps for the optional backends (`openai-whisper`, `openai`,
`faster-whisper`, `deep-translator`, `babel`) are all pinned only by name in
`requirements.txt`. Backends not used by current CLI flags may go
unimported; but they are still installed by `pip install -r
requirements.txt` so the user can swap backends without re-installing.
If you prune unused deps, do it via comments in `requirements.txt`,
not by removing lines — the readme still references both code paths.

Dubbing (`--dub`) adds `demucs` (source separation; pulls in torch/numpy/
soundfile) plus `numpy` + `soundfile` used directly by `assemble_and_mix()`.
The TTS backends need no Python package — they're stdlib-`urllib` HTTP — only
an env key: `ELEVENLABS_API_KEY` (default) or `OPENAI_API_KEY` (`--tts openai`).
System tools required at runtime for dubbing: `ffmpeg`/`ffprobe` (already
needed) and `demucs` on PATH (all guarded by `ensure_tool()`).
