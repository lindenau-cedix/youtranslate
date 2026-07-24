# youtranslate

Download a YouTube video in any language, transcribe it, translate the speech
to another language, and either **burn translated subtitles** into the video or
**dub it** — replace the original voices with generated, translated speech while
keeping the background audio (music, ambience, noise).

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
    ├──────────────► subtitles (--subtitles burn|soft)
    │   [4a] ffmpeg + libass    burns / attaches the translated SRT
    │
    └──────────────► dubbing (--dub)
        [4b] demucs            splits audio → vocals + background
             ElevenLabs / OpenAI TTS   synthesizes translated speech per segment
             ffmpeg atempo     time-fits each clip to its slot (A/V stays in sync)
             numpy mix         lays speech over the kept background, muxes it in
```

Subtitles and dubbing are **independent** — do either, both, or (with `--dub`)
audio-only. See [Dubbing](#dubbing-replace-the-voices-keep-the-background) below.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on your PATH (with `libass` enabled — bundled by default
  in most static builds). Install via your package manager:
  - Debian/Ubuntu:  `sudo apt-get install ffmpeg`
  - macOS (brew):  `brew install ffmpeg`
  - Windows:       https://www.gyan.dev/ffmpeg/builds/
- For GPU acceleration with `faster-whisper`, an NVIDIA GPU plus a working CUDA
  runtime is helpful but not required — CPU inference works, just slower.
- For **dubbing** (`--dub`), additionally:
  - `demucs` (installed by `requirements.txt`) for source separation. It pulls
    in PyTorch; a GPU makes separation much faster but CPU works.
  - A text-to-speech key in your environment:
    `export ELEVENLABS_API_KEY=...` (default), or `--tts openai` to reuse
    `OPENAI_API_KEY`.

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

## Dubbing: replace the voices, keep the background

`--dub` doesn't just put subtitles on the screen — it **replaces the spoken
audio** with generated, translated speech, while keeping the music, ambience and
noise from the original. Under the hood:

1. **Separate.** [Demucs](https://github.com/adefossez/demucs) splits the audio
   into a `vocals` stem (the original speech, which we discard) and a
   `no_vocals` stem (everything else, which we keep).
2. **Synthesize.** Each translated segment is spoken by a TTS backend
   (ElevenLabs by default) in the target language.
3. **Time-fit.** Translated lines rarely match the original length, so each clip
   is sped up just enough (capped by `--dub-max-tempo`, default 1.5×) to fit its
   original time slot — so lips and audio stay roughly in sync. Clips are never
   slowed down (that tends to drone).
4. **Mix & mux.** The fitted clips are laid onto a silent timeline at their
   original timestamps, mixed over the preserved background, and muxed back in as
   the video's new audio track. The video itself is stream-copied (no re-encode)
   unless you also burn subtitles.

```bash
export OPENAI_API_KEY=sk-...          # for transcription-translation (default)
export ELEVENLABS_API_KEY=...         # for the dubbed voice (default TTS)
python youtranslate.py "URL" --dub --target-lang de
# -> ./out/<title>.en.de.dubbed.mp4
```

By default `--dub` **also** burns subtitles (subtitles default to `burn`). For a
clean audio-only dub, turn them off:

```bash
python youtranslate.py "URL" --dub --subtitles none
```

### Keep the original speaker's voice (voice cloning)

With ElevenLabs you can go one step further and clone the *original speaker's
voice* from the separated vocals, so the dub sounds like the same person
speaking the new language:

```bash
python youtranslate.py "URL" --dub --clone-voice --target-lang de
```

This uses ElevenLabs Instant Voice Cloning (requires a paid ElevenLabs plan). A
temporary voice is created from a ~1-minute clean sample of the separated
vocals and **deleted afterwards** unless you pass `--keep-clone`.

> ⚠️ **Only clone voices you have the right to clone.** Cloning a real person's
> voice without consent may be illegal in your jurisdiction and violates the
> ElevenLabs terms of service. Use this on your own voice, public-domain
> material, or content you're licensed to dub.

### TTS backends

| `--tts` | Key | Notes |
|---|---|---|
| `elevenlabs` (default) | `ELEVENLABS_API_KEY` | ~100 languages; `--clone-voice` support; `--tts-voice` takes a voice ID or name (default *Rachel*); `--tts-model` default `eleven_multilingual_v2`. |
| `openai` | `OPENAI_API_KEY` | Reuses your translation key; `--tts-voice` is a preset like `alloy`/`nova`; `--tts-model` default `gpt-4o-mini-tts`. No voice cloning. |

### Mixing controls

- `--dub-bg-gain` (default `0.7`) — background loudness in the mix. Lower it if
  music drowns out the dialogue.
- `--dub-voice-gain` (default `1.0`) — dubbed-voice loudness.
- `--dub-max-tempo` (default `1.5`, max `2.0`) — how aggressively clips may be
  sped up to fit. Higher keeps sync tighter but can sound rushed.
- `--demucs-model` (default `htdemucs`) and `--dub-device` (`auto`/`cpu`/`cuda`)
  tune the separation step.

### HTTP 403: use your Chrome session

Some YouTube videos reject anonymous yt-dlp requests with `HTTP Error 403:
Forbidden`. If the video works while you are signed in to Google Chrome, let
yt-dlp read that browser profile's cookies:

```bash
python youtranslate.py "URL" --cookies-from-browser chrome
```

This passes yt-dlp's `--cookies-from-browser` option directly, so its complete
browser syntax is supported. To select a non-default Chrome profile, for
example, use `--cookies-from-browser "chrome:Profile 2"`. Close Chrome first if
yt-dlp reports that its cookie database is locked. Cookies stay in Chrome's
profile and are not copied into the output directory, but they grant access as
your signed-in account; do not share command output or browser profile data.
Use this only for videos you are allowed to access and download.

### Useful flags

| Flag | What it does | Default |
|---|---|---|
| `--cookies-from-browser` | Load cookies from a browser for yt-dlp (for example, `chrome`) | off |
| `--model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `small` |
| `--engine` | `openai-whisper` or `faster-whisper` | `openai-whisper` |
| `--source-lang` | Source language code (`en`, `fr`, `de`, `ja`, ...) or `auto` to let Whisper detect | `auto` |
| `--translator` | `openai` or `deep` (Google, free) | `openai` |
| `--target-lang` | Output language code (e.g. `de`, `fr`, `es`, `pt-BR`) | `de` |
| `--subtitles` | Subtitle handling: `burn`, `soft`, or `none` | `burn` |
| `--soft-subs` | Deprecated alias for `--subtitles soft` | off |
| `--burn-encoder` | Video encoder when burning: `auto`, `nvenc`, or `libx264` | `auto` |
| `--font` | Font name passed to libass (use `DejaVu Sans` on Linux) | `Arial` |
| `--font-size` | Pixel size of subtitle text | auto |
| `--margin-v` | Distance (px) from bottom of frame | `30` |
| `--dub` | Replace voices with translated speech, keep the background | off |
| `--tts` | TTS backend for `--dub`: `elevenlabs` or `openai` | `elevenlabs` |
| `--tts-voice` | Voice ID/name (ElevenLabs) or preset (OpenAI) | *Rachel* / `alloy` |
| `--tts-model` | TTS model id | `eleven_multilingual_v2` / `gpt-4o-mini-tts` |
| `--clone-voice` | Clone the original speaker's voice (ElevenLabs, paid) | off |
| `--keep-clone` | Keep the cloned voice in your account afterwards | off |
| `--keep-dub-audio` | Keep the dub work dir (stems, clips, mixed track) | off |
| `--demucs-model` | Demucs model for source separation | `htdemucs` |
| `--dub-device` | Device for separation: `auto`, `cpu`, `cuda` | `auto` |
| `--dub-max-tempo` | Max speed-up to fit a clip to its slot (≤ 2.0) | `1.5` |
| `--dub-bg-gain` / `--dub-voice-gain` | Background / voice loudness in the mix | `0.7` / `1.0` |

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

## GPU-accelerated subtitle burn (NVENC)

By default `--burn-encoder auto` picks `h264_nvenc` (NVIDIA's hardware H.264
encoder) when it's available, and falls back to `libx264` (CPU) otherwise.
NVENC typically gives **5–15× faster** encoding with no perceptible quality
loss versus the libx264 path. Audio is still `-c:a copy` in both cases.

### Check if your ffmpeg already supports NVENC

Most distro packages of ffmpeg (`apt install ffmpeg`, `brew install ffmpeg`)
are built **without** `--enable-nvenc`. Test yours:

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc
```

If you see `h264_nvenc` (and ideally `hevc_nvenc`) listed, you're done — the
script will use it automatically. If not, you'll need an ffmpeg that was built
with NVENC enabled.

### Build ffmpeg from source with `--enable-nvenc`

On Debian / Ubuntu (tested against 22.04 and 24.04). The build takes ~10–20 min
and produces a static-ish `ffmpeg`/`ffprobe` in `/usr/local/bin`.

1. **Install the NVIDIA driver** if you haven't already. You only need the
   proprietary driver, not the full CUDA toolkit:

   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-560   # or whatever `ubuntu-drivers devices` recommends
   sudo reboot
   nvidia-smi                                 # should show your GPU
   ```

2. **Install build dependencies.** `libnvidia-encode-...` headers are the
   critical ones — without them, `./configure` will silently skip NVENC:

   ```bash
   sudo apt-get install -y \
       build-essential pkg-config yasm nasm \
       libx264-dev libx265-dev libvpx-dev libmp3lame-dev libopus-dev libvorbis-dev \
       libfdk-aac-dev libass-dev libfreetype-dev libfontconfig1-dev \
       libnvidia-encode-560 libnvidia-decode-560 libnvidia-utils-560 \
       libtool libssl-dev
   ```

   The package suffix (`560`) should match the driver version you installed in
   step 1. If `apt` doesn't find `libnvidia-encode-*`, install
   `nvidia-cuda-toolkit` or pick the version that matches your driver.

3. **Get the ffmpeg source.** Latest stable as of writing is 7.x:

   ```bash
   cd /tmp
   wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.xz
   tar xf ffmpeg-7.1.tar.xz
   cd ffmpeg-7.1
   ```

4. **Configure with NVENC enabled:**

   ```bash
   ./configure \
       --enable-gpl \
       --enable-nonfree \
       --enable-cuda-nvcc \
       --enable-libnvenc \
       --enable-libx264 \
       --enable-libx265 \
       --enable-libvpx \
       --enable-libmp3lame \
       --enable-libopus \
       --enable-libvorbis \
       --enable-libfdk-aac \
       --enable-libass \
       --enable-libfreetype \
       --enable-libfontconfig \
       --extra-cflags=-I/usr/local/cuda/include \
       --extra-ldflags=-L/usr/local/cuda/lib64
   ```

   The two important flags are `--enable-cuda-nvcc` and `--enable-libnvenc`.
   Without them, NVENC will not appear in the encoder list. If `./configure`
   prints "WARNING: libnvenc not found" at the end, your
   `libnvidia-encode-*` package is missing or the version doesn't match your
   driver — fix that before building.

5. **Build and install:**

   ```bash
   make -j"$(nproc)"
   sudo make install
   sudo ldconfig
   ```

6. **Verify:**

   ```bash
   hash -r                                 # refresh PATH cache
   which ffmpeg                            # should be /usr/local/bin/ffmpeg
   ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc
   # expect:
   #  V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
   #  V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
   nvidia-smi                               # confirm driver still loads
   ```

### macOS and Windows

- **macOS:** Apple's hardware H.264 encoder is exposed via VideoToolbox. ffmpeg
  can use it as `h264_videotoolbox`, which `burn_subtitles()` does not select
  automatically. To use it, edit the encoder block in `burn_subtitles()` to
  add an `elif encoder == "videotoolbox":` branch with `-c:v h264_videotoolbox
  -q:v 22`, or use `--burn-encoder libx264`. Apple Silicon Macs work; Intel
  Macs need a CPU with QuickSync.
- **Windows:** Build or download a BtbN NVENC-enabled ffmpeg from
  https://github.com/BtbN/ffmpeg-builds — pick a `shared` or `git` build with
  `nvenc` in the filename. Drop the two `.exe` files somewhere on your PATH.

### Usage

```bash
# Let the script auto-pick (uses NVENC if ffmpeg supports it, else libx264):
python youtranslate.py "URL"

# Force NVENC (errors out if unavailable):
python youtranslate.py "URL" --burn-encoder nvenc

# Force CPU libx264 (e.g., for benchmarking or bit-exact reproducibility):
python youtranslate.py "URL" --burn-encoder libx264
```

The script prints which encoder it's using before invoking ffmpeg, so you'll
see `using NVIDIA NVENC encoder (h264_nvenc)` or `using CPU encoder (libx264)`
on stderr-equivalent output.

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
  the source. If you need bit-exact video, pass `--subtitles soft` to attach
  subtitles as a non-re-encoding stream instead. (`--dub` alone also stream-copies
  the video — only the audio changes.)
- **Dubbing quality depends on separation.** Demucs is very good but not perfect;
  faint traces of the original voice can remain in the background stem, and dense
  music can leave artifacts. This is inherent to source separation.
- **Lip-sync is approximate.** Speech is fitted to each segment's timing by
  speeding it up within `--dub-max-tempo`, not by matching mouth movements. It
  tracks the original pacing closely but is not frame-accurate lip-sync.
- **Dubbing costs add up.** Every segment is a TTS call. ElevenLabs bills per
  character and voice cloning needs a paid plan; `--tts openai` bills per
  character too. Transcription/translation costs are unchanged.
- **Cloning consent.** `--clone-voice` reproduces a real person's voice. Only use
  it where you have permission — see the warning in the Dubbing section.
