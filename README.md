# youtranslate

Download a YouTube video in any language, transcribe it, translate the speech
to another language, and burn the translated subtitles into the video.

## Pipeline

```
YouTube URL
    │
    ▼
[1] yt-dlp              downloads best video+audio, muxed to mp4
    │
    ▼
[2] openai-whisper      transcribes audio → segments in the source language
    │
    ▼
[3] OpenAI Chat API     gpt-4o-mini, per segment → target-language segments
    │
    ▼
[4] ffmpeg + libass     burns the translated SRT into a copy of the video
```

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on your PATH (with `libass` enabled — bundled by default
  in most static builds). Install via your package manager:
  - Debian/Ubuntu:  `sudo apt-get install ffmpeg`
  - macOS (brew):  `brew install ffmpeg`
  - Windows:       https://www.gyan.dev/ffmpeg/builds/
- For GPU acceleration with `faster-whisper`, an NVIDIA GPU plus a working CUDA
  runtime is helpful but not required — CPU inference works, just slower.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Both transcription and translation default to OpenAI, so set your API key first:

```bash
export OPENAI_API_KEY=sk-...
python youtranslate.py "https://www.youtube.com/watch?v=VIDEO_ID" --output ./out
```

That produces `./out/<title>.en.de.subtitled.mp4` — the original video with German
subtitles hardcoded on the bottom. The `<src>.<tgt>` naming reflects the source
language Whisper detected (or the `--source-lang` you pinned) and the target
language you asked for.

If you'd like to keep the subtitle files alongside the video:

```bash
python youtranslate.py "URL" --output ./out --keep-srt
```

This will additionally produce `<title>.en.srt` (original) and
`<title>.en.de.srt` (translated).

### Useful flags

| Flag | What it does | Default |
|---|---|---|
| `--model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `small` |
| `--engine` | `openai-whisper` or `faster-whisper` | `openai-whisper` |
| `--source-lang` | Source language code (`en`, `fr`, `de`, `ja`, ...) or `auto` to let Whisper detect | `auto` |
| `--translator` | `openai` or `deep` (Google, free) | `openai` |
| `--target-lang` | Output subtitle language code (e.g. `de`, `fr`, `es`, `pt-BR`) | `de` |
| `--soft-subs` | Attach subtitles as a separate (toggleable) stream instead of burning them | off |
| `--font` | Font name passed to libass (use `DejaVu Sans` on Linux) | `Arial` |
| `--font-size` | Pixel size of subtitle text | auto |
| `--margin-v` | Distance (px) from bottom of frame | `30` |

### Examples

Use the larger, more accurate Whisper model on a long lecture:

```bash
python youtranslate.py "URL" --model medium
```

Translate a French video to Spanish (pin the source language):

```bash
python youtranslate.py "URL" --source-lang fr --target-lang es
```

Let Whisper detect the source language and translate to French:

```bash
python youtranslate.py "URL" --target-lang fr
```

Skip OpenAI entirely and use the free local transcription + Google Translate:

```bash
pip install faster-whisper deep-translator
python youtranslate.py "URL" --engine faster-whisper --translator deep
```

Embed the subtitles as a soft (toggleable) track:

```bash
python youtranslate.py "URL" --soft-subs
```

## Notes & caveats

- **YouTube terms of service.** Only download videos you have the right to
  download. Don't redistribute copyrighted material.
- **Speed.** End-to-end runtime is dominated by Whisper transcription. A 10-min
  video takes ~3 min on CPU with `--model small`. Bigger models (medium, large-v3)
  give noticeably better segmentation at the cost of runtime.
- **Translation by segment.** Each subtitle line is translated independently.
  This preserves timing faithfully but can occasionally lose a little context;
  that's a deliberate trade-off — the alternative (translate the whole transcript
  first, then re-time) is much less accurate.
- **Burning subtitles is lossy.** A re-encoded video uses H.264 at CRF 22, which
  is visually transparent at typical viewing sizes but is *not* the same bytes as
  the source. If you need bit-exact video, pass `--soft-subs` to attach subtitles
  as a non-re-encoding stream instead.
