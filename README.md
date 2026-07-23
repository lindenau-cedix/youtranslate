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
| `--target-lang` | Output subtitle language code (e.g. `de`, `fr`, `es`, `pt-BR`) | `de` |
| `--soft-subs` | Attach subtitles as a separate (toggleable) stream instead of burning them | off |
| `--burn-encoder` | Video encoder for the subtitled output: `auto`, `nvenc`, or `libx264` | `auto` |
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
  the source. If you need bit-exact video, pass `--soft-subs` to attach subtitles
  as a non-re-encoding stream instead.
