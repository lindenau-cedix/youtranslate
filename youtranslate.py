#!/usr/bin/env python3
"""
youtranslate — Download a YouTube video in any language, transcribe it,
translate the speech to another language, and either burn translated
subtitles into the video **or dub it** (replace the original voices with
generated, translated speech while keeping the background audio).

Pipeline:
    1. yt-dlp downloads the video (best video + best audio, muxed to mp4).
    2. faster-whisper (or openai-whisper) transcribes the audio to segments
       in the source language.
    3. deep-translator (Google, free) or OpenAI translates each segment to the
       target language.
    4a. Subtitles: ffmpeg + libass burns the translated SRT into a copy of the
        video (or attaches it as a soft mov_text track).
    4b. Dubbing (--dub): Demucs splits the audio into vocals + background,
        ElevenLabs (or OpenAI) synthesises the translated speech per segment,
        each clip is time-fitted to its slot, mixed over the preserved
        background, and muxed back in as the new audio track.

Subtitles and dubbing are independent — enable either, both, or neither
(at least one must be active).

Outputs (in --output dir):
    <title>.<src>.<tgt>.dubbed.mp4              — video with replaced audio.
    <title>.<src>.<tgt>.subtitled.mp4          — video with burned/soft subtitles.
    <title>.<src>.<tgt>.dubbed.subtitled.mp4   — both at once.
    <title>.<src>.srt / .<src>.<tgt>.srt       — subtitle files (with --keep-srt).

Example:
    python youtranslate.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \\
        --output ./out --model small --source-lang en --target-lang de --dub
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# --- Data -------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A timed text segment from transcription / translation."""
    start: float  # seconds
    end: float    # seconds
    text: str


# --- Helpers ----------------------------------------------------------------


def ensure_tool(name: str) -> None:
    """Verify an external CLI tool is on PATH; otherwise exit with a hint."""
    if shutil.which(name) is None:
        raise SystemExit(
            f"required CLI tool '{name}' not found on PATH. "
            f"Install it and try again."
        )


def run_checked(cmd: Sequence[str], what: str) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing stderr and exiting with a message on failure.

    Used by the dubbing stage and the muxing helpers; the older download /
    transcribe code keeps its own bespoke error messages.
    """
    proc = subprocess.run(list(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout or "")
        raise SystemExit(
            f"{cmd[0]} failed during {what} (exit code {proc.returncode})."
        )
    return proc


def fmt_srt_time(t: float) -> str:
    """Format seconds as `HH:MM:SS,mmm` SRT timestamp."""
    if t < 0:
        t = 0.0
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t - hours * 3600 - minutes * 60
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    # Edge case: rounding can push millis to 1000.
    if millis >= 1000:
        millis = 0
        whole += 1
        if whole >= 60:
            whole = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{whole:02d},{millis:03d}"


def wrap_text(text: str, width: int = 80) -> str:
    """Word-wrap text to roughly `width` chars per line, preserving explicit newlines."""
    out_lines: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            out_lines.append(line)
            continue
        words = line.split()
        cur: List[str] = []
        cur_len = 0
        for w in words:
            new_len = cur_len + (1 if cur else 0) + len(w)
            if new_len > width and cur:
                out_lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len = new_len
        if cur:
            out_lines.append(" ".join(cur))
    return "\n".join(out_lines)


def human_lang_name(code: str) -> str:
    """Resolve an ISO-639 (or BCP-47) code to its English display name.

    Examples: "fr" → "French", "pt-BR" → "Brazilian Portuguese", "en" → "English".
    Falls back to the raw code if babel can't resolve it (e.g. user passes a
    custom tag).
    """
    try:
        from babel import Locale
        from babel.core import UnknownLocaleError
    except ImportError:
        return code
    try:
        return Locale.parse(code).get_display_name("en")
    except (UnknownLocaleError, ValueError):
        return code


def iso1(code: str) -> str:
    """Reduce a language code to its bare ISO-639-1 form (`pt-BR` → `pt`)."""
    return code.split("-")[0].split("_")[0].lower()


# --- Step 1: download -------------------------------------------------------


def download(
    url: str, out_dir: Path, cookies_from_browser: Optional[str] = None
) -> Path:
    """Download the given YouTube URL into out_dir; return the resulting file path."""
    ensure_tool("yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--merge-output-format", "mp4",
        "-f", "bv*+ba/b",
    ]
    if cookies_from_browser:
        cmd.extend(["--cookies-from-browser", cookies_from_browser])
    cmd.extend([
        "-o", str(out_dir / "%(title)s.%(ext)s"),
        url,
    ])
    print("  $ " + subprocess.list2cmdline(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # yt-dlp prints useful diagnostics to stderr when it fails.
        diagnostics = proc.stderr or proc.stdout or ""
        sys.stderr.write(diagnostics)
        if not cookies_from_browser and re.search(
            r"(?:HTTP Error )?403|Forbidden", diagnostics, re.IGNORECASE
        ):
            if diagnostics and not diagnostics.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.write(
                "Hint: retry with `--cookies-from-browser chrome` to let yt-dlp "
                "use your signed-in Chrome session.\n"
            )
        raise SystemExit(f"yt-dlp failed with exit code {proc.returncode}")

    # yt-dlp prints "Destination: <path>" on success — try to parse it first.
    for line in (proc.stdout or "").splitlines()[::-1]:
        m = re.search(r"\[Destination\] (.+)$", line) or re.search(
            r"\[Merged\] (.+)$", line
        )
        if m:
            candidate = Path(m.group(1).strip())
            if candidate.exists():
                return candidate

    # Fallback: pick the most recently modified video file in the directory.
    candidates = sorted(
        (p for p in out_dir.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if p.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return p
    raise SystemExit("yt-dlp finished but no video file was produced.")


# --- Step 2: transcribe -----------------------------------------------------


def transcribe_faster_whisper(
    video_path: Path, model_size: str, source_lang: str
) -> tuple[List[Segment], str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "faster-whisper is not installed. "
            "Install with `pip install faster-whisper`, or pass --engine openai-whisper."
        ) from e

    # device='auto' picks CUDA when available, else CPU. compute_type='auto' picks a
    # good precision for the chosen device (float16 on GPU, int8 on CPU by default).
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    # When source_lang is "auto", drop the language kwarg so Whisper runs its
    # own detection on the first 30s. Otherwise pin the language explicitly.
    # task="transcribe" — we want the source-language text; the per-segment
    # translator (deep / openai) handles the language conversion.
    transcribe_kwargs = {"task": "transcribe"}
    if source_lang != "auto":
        transcribe_kwargs["language"] = source_lang

    segments_iter, info = model.transcribe(str(video_path), **transcribe_kwargs)
    print(f"  detected language: {info.language} "
          f"(probability {info.language_probability:.2f})")

    out: List[Segment] = []
    for seg in segments_iter:
        out.append(Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip()))
    return out, info.language


def transcribe_openai_whisper(
    video_path: Path, model_size: str, source_lang: str
) -> tuple[List[Segment], str]:
    try:
        import whisper  # openai-whisper
    except ImportError as e:
        raise SystemExit(
            "openai-whisper is not installed. "
            "Install with `pip install openai-whisper`."
        ) from e

    model = whisper.load_model(model_size)
    transcribe_kwargs = {"task": "transcribe"}
    if source_lang != "auto":
        transcribe_kwargs["language"] = source_lang
    result = model.transcribe(str(video_path), **transcribe_kwargs)

    out: List[Segment] = []
    for seg in result.get("segments", []):
        out.append(
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=seg["text"].strip(),
            )
        )
    detected = result.get("language", source_lang)
    return out, detected


def transcribe(
    video_path: Path, model_size: str, engine: str, source_lang: str
) -> tuple[List[Segment], str]:
    if engine == "faster-whisper":
        return transcribe_faster_whisper(video_path, model_size, source_lang)
    elif engine == "openai-whisper":
        return transcribe_openai_whisper(video_path, model_size, source_lang)
    raise SystemExit(f"unknown engine '{engine}'")


# --- Step 3: translate ------------------------------------------------------


def translate_segments_deep(
    segments: Sequence[Segment], target_lang: str = "de"
) -> List[Segment]:
    try:
        from deep_translator import GoogleTranslator
    except ImportError as e:
        raise SystemExit(
            "deep-translator is not installed. "
            "Install with `pip install deep-translator`, or pass --translator openai."
        ) from e

    # GoogleTranslator is a thin client over the public Google Translate endpoint;
    # it can rate-limit on long videos, so we translate per-segment (slow but
    # stable, preserves timing).
    translator = GoogleTranslator(source="en", target=target_lang)

    out: List[Segment] = []
    total = len(segments)
    for i, seg in enumerate(segments, start=1):
        if not seg.text:
            out.append(Segment(seg.start, seg.end, ""))
            continue
        translated = ""
        for attempt in range(3):
            try:
                translated = translator.translate(seg.text) or ""
                break
            except Exception as e:  # noqa: BLE001 — deep-translator raises wide.
                if attempt == 2:
                    raise SystemExit(
                        f"translation failed on segment {i}/{total}: {e}"
                    ) from e
                import time
                time.sleep(1.5 * (attempt + 1))
        print(f"  translated {i}/{total}", end="\r")
        out.append(Segment(seg.start, seg.end, translated.strip()))
    print()  # newline after the carriage-return progress line
    return out


def translate_segments_openai(
    segments: Sequence[Segment], target_lang_human: str
) -> List[Segment]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Either set it or use --translator deep."
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "openai is not installed. `pip install openai` or use --translator deep."
        ) from e

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")

    out: List[Segment] = []
    total = len(segments)
    system = "You translate concise subtitle lines. Output only the translation, no quotes."
    for i, seg in enumerate(segments, start=1):
        if not seg.text:
            out.append(Segment(seg.start, seg.end, ""))
            continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Translate the next subtitle line to {target_lang_human}. "
                        f"Keep it short, natural, and on one or two lines.\n\n"
                        f"{seg.text}"
                    ),
                },
            ],
            temperature=0.0,
        )
        translated = (resp.choices[0].message.content or "").strip()
        print(f"  translated {i}/{total}", end="\r")
        out.append(Segment(seg.start, seg.end, translated))
    print()
    return out


def translate_segments(
    segments: Sequence[Segment], translator: str, target_lang: str,
    target_lang_human: str,
) -> List[Segment]:
    if translator == "deep":
        return translate_segments_deep(segments, target_lang=target_lang)
    if translator == "openai":
        return translate_segments_openai(segments, target_lang_human=target_lang_human)
    raise SystemExit(f"unknown translator '{translator}'")


# --- Step 4: subtitle I/O & burn --------------------------------------------


def write_srt(segments: Sequence[Segment], path: Path) -> None:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_time(seg.start)} --> {fmt_srt_time(seg.end)}")
        lines.append(wrap_text(seg.text))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def ffprobe_dimensions(video_path: Path) -> tuple[int, int]:
    ensure_tool("ffprobe")
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ],
        text=True,
    ).strip()
    w, h = out.split("x")
    return int(w), int(h)


def ffmpeg_has_encoder(name: str) -> bool:
    """Return True if the system ffmpeg binary advertises the named encoder.

    Used to detect NVENC support (`h264_nvenc`, `hevc_nvenc`) at runtime so the
    script can fall back to libx264 when GPU encoding isn't available. ffmpeg
    lists every supported encoder on stderr when `-encoders` is passed.
    """
    ensure_tool("ffmpeg")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    )
    # Each line of `ffmpeg -encoders` is like: " V..... = Video codec ..." then
    # " V....D h264_nvenc ..." The encoder name appears as the second column.
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def resolve_encoder(encoder: str) -> bool:
    """Map the --burn-encoder choice to a use_nvenc bool, printing the pick.

    Shared by burn_subtitles() so the NVENC/libx264 decision lives in one place.
    """
    if encoder == "auto":
        use_nvenc = ffmpeg_has_encoder("h264_nvenc")
        if use_nvenc:
            print("  using NVIDIA NVENC encoder (h264_nvenc)")
        else:
            print("  using CPU encoder (libx264) — "
                  "h264_nvenc not found in this ffmpeg build")
        return use_nvenc
    if encoder == "nvenc":
        if not ffmpeg_has_encoder("h264_nvenc"):
            raise SystemExit(
                "--burn-encoder nvenc requested, but this ffmpeg build does not "
                "advertise h264_nvenc. See README for instructions on building "
                "ffmpeg with --enable-nvenc."
            )
        print("  using NVIDIA NVENC encoder (h264_nvenc)")
        return True
    if encoder == "libx264":
        print("  using CPU encoder (libx264)")
        return False
    raise SystemExit(f"unknown --burn-encoder '{encoder}'")


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    out_path: Path,
    *,
    font: str = "Arial",
    font_size: Optional[int] = None,
    margin_v: int = 30,
    encoder: str = "auto",
    audio_path: Optional[Path] = None,
) -> None:
    """Burn SRT subtitles into a copy of the video using ffmpeg + libass.

    `encoder` selects the video encoder:
      - "auto"    : use h264_nvenc if ffmpeg was built with NVENC, else libx264.
      - "nvenc"   : force h264_nvenc (errors out if ffmpeg doesn't support it).
      - "libx264" : force the CPU encoder.

    `audio_path` (used by the dubbing stage): when given, the video's original
    audio is dropped and this track is muxed in instead (re-encoded to AAC).
    When None, the source audio is stream-copied as before.
    """
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    width, height = ffprobe_dimensions(video_path)
    # Auto-pick a compact font size based on resolution.  The previous ratio
    # produced oversized subtitles on high-resolution videos; this keeps the
    # default roughly 3–4x smaller while remaining readable on smaller frames.
    if font_size is None:
        font_size = max(10, min(24, int(height / 100)))

    # libass force_style — names follow ASS spec. Alignment=2 = bottom-center.
    force_style = (
        f"FontName={font},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF&,"   # white
        f"OutlineColour=&H00000000&,"   # black outline
        f"BackColour=&H80000000&,"      # semi-transparent back box
        f"BorderStyle=1,"
        f"Outline=2,"
        f"Shadow=0,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )
    vf = f"subtitles={srt_path}:si=0:force_style='{force_style}'"

    # NVENC gives 5–15× speedup over libx264 with no perceptual quality loss;
    # libx264 is the universal fallback.
    use_nvenc = resolve_encoder(encoder)

    # Input / mapping / audio-codec depend on whether we're replacing the audio
    # with a dubbed track.
    inputs = ["-i", str(video_path)]
    if audio_path is not None:
        inputs += ["-i", str(audio_path)]
        maps = ["-map", "0:v:0", "-map", "1:a:0"]
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
    else:
        maps = []  # let ffmpeg pick the default video+audio streams
        audio_args = ["-c:a", "copy"]

    if use_nvenc:
        # NVENC: `-rc vbr -cq N -b:v 0` is the constant-quality mode analogous
        # to libx264's CRF. `-preset p4` ≈ libx264's "medium". `-hwaccel cuda`
        # decodes on the GPU too, shaving more time off large videos.
        head = ["ffmpeg", "-y", "-hwaccel", "cuda", *inputs, "-vf", vf]
        video_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "22",
            "-b:v", "0",
        ]
    else:
        head = ["ffmpeg", "-y", *inputs, "-vf", vf]
        video_args = [
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
        ]
    cmd = [
        *head, *maps, *video_args, *audio_args,
        "-movflags", "+faststart", str(out_path),
    ]
    print("  $ ffmpeg " + " ".join(cmd[1:]))
    run_checked(cmd, "subtitle burn")


def attach_or_replace(
    video_path: Path,
    out_path: Path,
    *,
    audio_path: Optional[Path] = None,
    soft_srt: Optional[Path] = None,
    target_lang: Optional[str] = None,
) -> None:
    """Mux without re-encoding the video (video is stream-copied).

    - audio_path given → replace the audio with it (encoded to AAC); this is the
      dubbing-without-subtitles path.
    - audio_path None  → keep the source audio (stream copy).
    - soft_srt given   → attach it as a toggleable mov_text subtitle stream.

    Handles every non-burn combination: dub-only, soft-subs-only, dub+soft-subs.
    """
    ensure_tool("ffmpeg")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    next_input = 1
    audio_input = None
    srt_input = None
    if audio_path is not None:
        cmd += ["-i", str(audio_path)]
        audio_input = next_input
        next_input += 1
    if soft_srt is not None:
        cmd += ["-i", str(soft_srt)]
        srt_input = next_input
        next_input += 1

    cmd += ["-map", "0:v:0"]
    if audio_input is not None:
        cmd += ["-map", f"{audio_input}:a:0"]
    else:
        cmd += ["-map", "0:a:0?"]  # '?' → don't fail if the source has no audio
    if srt_input is not None:
        cmd += ["-map", f"{srt_input}:0"]

    cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac" if audio_input is not None else "copy"]
    if srt_input is not None:
        cmd += ["-c:s", "mov_text"]
        if target_lang:
            cmd += ["-metadata:s:s:0", f"language={target_lang}"]
    cmd += ["-movflags", "+faststart", str(out_path)]

    print("  $ ffmpeg " + " ".join(cmd[1:]))
    run_checked(cmd, "muxing")


# --- Step 5: dub (voice replacement) ----------------------------------------
#
# Replaces the original spoken audio with generated, translated speech while
# keeping music / ambience / noise. The original speech is removed by source
# separation (Demucs two-stems), the translated speech is synthesised per
# segment (ElevenLabs / OpenAI), time-fitted so A/V stays in sync, and mixed
# back over the preserved background.

DUB_SR = 44100          # Demucs operates at 44.1 kHz; we keep everything there.
DUB_CHANNELS = 2
ELEVENLABS_API = "https://api.elevenlabs.io/v1"
# "Rachel" — a stock multilingual ElevenLabs voice. Any voice ID or name works
# via --tts-voice; this is only the default when the user doesn't pick one.
DEFAULT_ELEVEN_VOICE = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_ELEVEN_MODEL = "eleven_multilingual_v2"
DEFAULT_OPENAI_TTS_VOICE = "alloy"
DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"


@dataclass
class DubConfig:
    """Everything the dubbing stage needs, gathered from the CLI in one place."""
    engine: str            # "elevenlabs" | "openai"
    api_key: str
    voice: str             # ElevenLabs voice ID/name, or OpenAI voice name
    model: str             # TTS model id
    clone_voice: bool      # ElevenLabs Instant Voice Clone from the original speaker
    keep_clone: bool       # keep the cloned voice in your account afterwards
    demucs_model: str
    device: str            # "auto" | "cpu" | "cuda"
    max_tempo: float       # cap on speed-up applied to fit a slot
    bg_gain: float
    voice_gain: float


def _model_accepts_language_code(model: str) -> bool:
    """ElevenLabs turbo/flash/v3 models accept an explicit language_code."""
    m = model.lower()
    return any(k in m for k in ("turbo", "flash", "v2_5", "v3"))


def _looks_like_voice_id(s: str) -> bool:
    """ElevenLabs voice IDs are 20-char alphanumerics; names generally aren't."""
    return bool(re.fullmatch(r"[A-Za-z0-9]{20}", s))


def _http_bytes(
    url: str, payload: dict, headers: dict, *, accept: Optional[str] = None,
    service: str = "API",
) -> bytes:
    """POST JSON, return the raw response bytes (audio). Exit clearly on error."""
    import json
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if accept:
        req.add_header("Accept", accept)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"{service} error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"{service} request failed: {e}") from e


def _multipart(fields: dict, files: Sequence[Tuple[str, str, bytes, str]]) -> Tuple[bytes, str]:
    """Build a multipart/form-data body. files: (name, filename, bytes, content_type)."""
    boundary = "----youtranslateBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    for name, filename, content, ctype in files:
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += content
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), boundary


def elevenlabs_resolve_voice(name_or_id: str, api_key: str) -> str:
    """Return a voice ID for a voice ID or a voice *name* (looked up via API)."""
    if _looks_like_voice_id(name_or_id):
        return name_or_id
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"{ELEVENLABS_API}/voices", method="GET")
    req.add_header("xi-api-key", api_key)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"ElevenLabs voice lookup failed ({e.code}). "
            f"Pass a 20-character voice ID via --tts-voice to skip the lookup."
        ) from e
    for v in data.get("voices", []):
        if v.get("name", "").lower() == name_or_id.lower():
            return v["voice_id"]
    raise SystemExit(
        f"ElevenLabs voice '{name_or_id}' not found in your account. "
        f"Use a voice ID or an exact voice name."
    )


def synthesize_elevenlabs(
    text: str, out_path: Path, *, api_key: str, voice_id: str, model: str,
    lang_code: Optional[str] = None,
) -> Path:
    """Synthesise one line with ElevenLabs; write mp3 bytes to out_path."""
    import urllib.parse

    url = (
        f"{ELEVENLABS_API}/text-to-speech/{urllib.parse.quote(voice_id)}"
        f"?output_format=mp3_44100_128"
    )
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    if lang_code and _model_accepts_language_code(model):
        payload["language_code"] = lang_code
    audio = _http_bytes(
        url, payload, {"xi-api-key": api_key},
        accept="audio/mpeg", service="ElevenLabs",
    )
    out_path.write_bytes(audio)
    return out_path


def synthesize_openai(
    text: str, out_path: Path, *, api_key: str, voice: str, model: str,
) -> Path:
    """Synthesise one line with the OpenAI speech API; write mp3 bytes."""
    payload = {"model": model, "voice": voice, "input": text, "response_format": "mp3"}
    audio = _http_bytes(
        "https://api.openai.com/v1/audio/speech", payload,
        {"Authorization": f"Bearer {api_key}"},
        accept="audio/mpeg", service="OpenAI TTS",
    )
    out_path.write_bytes(audio)
    return out_path


def synthesize_segment(
    cfg: DubConfig, text: str, out_path: Path, *, voice_id: str,
    lang_code: Optional[str],
) -> Path:
    if cfg.engine == "elevenlabs":
        return synthesize_elevenlabs(
            text, out_path, api_key=cfg.api_key, voice_id=voice_id,
            model=cfg.model, lang_code=lang_code,
        )
    if cfg.engine == "openai":
        return synthesize_openai(
            text, out_path, api_key=cfg.api_key, voice=cfg.voice, model=cfg.model,
        )
    raise SystemExit(f"unknown --tts engine '{cfg.engine}'")


def elevenlabs_clone_voice(sample_path: Path, api_key: str, name: str) -> str:
    """Instant Voice Clone from a sample clip; return the new voice_id.

    Requires an ElevenLabs plan with voice cloning. Only ever call this on
    voices you have the right to clone.
    """
    import json
    import urllib.error
    import urllib.request

    content = Path(sample_path).read_bytes()
    body, boundary = _multipart(
        {"name": name, "remove_background_noise": "false"},
        [("files", "sample.wav", content, "audio/wav")],
    )
    req = urllib.request.Request(f"{ELEVENLABS_API}/voices/add", data=body, method="POST")
    req.add_header("xi-api-key", api_key)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(
            f"ElevenLabs voice cloning failed ({e.code}): {detail}\n"
            f"Voice cloning requires a paid ElevenLabs plan."
        ) from e
    voice_id = data.get("voice_id")
    if not voice_id:
        raise SystemExit("ElevenLabs cloning returned no voice_id.")
    return voice_id


def elevenlabs_delete_voice(voice_id: str, api_key: str) -> None:
    """Best-effort cleanup of a cloned voice; never raises."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"{ELEVENLABS_API}/voices/{voice_id}", method="DELETE")
    req.add_header("xi-api-key", api_key)
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass


def extract_audio(video_path: Path, out_wav: Path) -> Path:
    """Extract the video's audio to a 44.1 kHz stereo wav for separation."""
    ensure_tool("ffmpeg")
    run_checked(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-ac", str(DUB_CHANNELS), "-ar", str(DUB_SR),
         "-c:a", "pcm_s16le", str(out_wav)],
        "audio extraction",
    )
    return out_wav


def separate_vocals(
    audio_wav: Path, out_dir: Path, *, model: str = "htdemucs", device: str = "auto",
) -> Tuple[Path, Path]:
    """Split audio into (vocals, background) with Demucs two-stems mode.

    Returns (vocals_wav, no_vocals_wav). `no_vocals` keeps music, ambience and
    noise; `vocals` is the original speech we replace (and, with --clone-voice,
    the reference for cloning the speaker).
    """
    ensure_tool("demucs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["demucs", "--two-stems", "vocals", "-n", model, "-o", str(out_dir)]
    if device != "auto":
        cmd += ["-d", device]
    cmd += [str(audio_wav)]
    print("  $ " + subprocess.list2cmdline(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout or "")
        raise SystemExit(f"demucs failed with exit code {proc.returncode}.")
    stem_dir = out_dir / model / audio_wav.stem
    vocals = stem_dir / "vocals.wav"
    background = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        raise SystemExit(f"demucs finished but stems were not found under {stem_dir}.")
    return vocals, background


def probe_duration(path: Path) -> float:
    """Return a media file's duration in seconds (0.0 if unknown)."""
    ensure_tool("ffprobe")
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        text=True,
    ).strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def plan_tempo(natural: float, slot: float, max_tempo: float) -> float:
    """Speed factor to fit `natural` seconds of speech into a `slot`.

    Only ever speeds up (never slows down — that tends to drone): returns 1.0
    when the clip already fits, otherwise natural/slot capped at max_tempo. A
    capped clip will slightly overrun its slot; the assembly step tolerates that.
    """
    if slot <= 0 or natural <= 0:
        return 1.0
    if natural <= slot:
        return 1.0
    return min(natural / slot, max_tempo)


def fit_clip(
    in_path: Path, out_wav: Path, target_dur: float, *, max_tempo: float,
) -> float:
    """Re-time a synthesized clip to ~target_dur and normalise it to DUB_SR/stereo.

    Returns the tempo factor applied (for logging). atempo caps at 2.0 per
    filter instance, which is why max_tempo is clamped to 2.0 by the CLI.
    """
    natural = probe_duration(in_path)
    tempo = plan_tempo(natural, target_dur, max_tempo)
    af = []
    if abs(tempo - 1.0) > 0.01:
        af.append(f"atempo={tempo:.6f}")
    af.append(f"aresample={DUB_SR}")
    run_checked(
        ["ffmpeg", "-y", "-i", str(in_path),
         "-filter:a", ",".join(af),
         "-ac", str(DUB_CHANNELS), "-ar", str(DUB_SR),
         "-c:a", "pcm_s16le", str(out_wav)],
        "clip fitting",
    )
    return tempo


def assemble_and_mix(
    background_wav: Path,
    placements: Sequence[Tuple[float, Path]],
    out_wav: Path,
    *,
    bg_gain: float,
    voice_gain: float,
) -> Path:
    """Lay each fitted clip onto a silent timeline at its start time and mix it
    over the background. Overlapping clips sum; the result is peak-normalised to
    avoid clipping. `background_wav` must already be DUB_SR / stereo.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        raise SystemExit(
            "dubbing needs numpy + soundfile (installed with demucs). "
            "Run `pip install numpy soundfile`."
        ) from e

    bg, _ = sf.read(str(background_wav), dtype="float32", always_2d=True)
    n_samples, n_ch = bg.shape
    dub = np.zeros((n_samples, n_ch), dtype="float32")

    for start_s, clip_path in placements:
        clip, _ = sf.read(str(clip_path), dtype="float32", always_2d=True)
        if clip.shape[1] != n_ch:
            clip = (np.repeat(clip, n_ch, axis=1) if clip.shape[1] == 1
                    else clip[:, :n_ch])
        i0 = int(round(start_s * DUB_SR))
        if i0 >= n_samples:
            continue
        i1 = min(n_samples, i0 + clip.shape[0])
        dub[i0:i1] += clip[: i1 - i0]

    mixed = bg * bg_gain + dub * voice_gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    sf.write(str(out_wav), mixed, DUB_SR, subtype="PCM_16")
    return out_wav


def dub_audio(
    video_path: Path,
    tgt_segments: Sequence[Segment],
    *,
    target_lang: str,
    work: Path,
    cfg: DubConfig,
) -> Path:
    """Produce a dubbed audio track: translated speech over the kept background.

    Returns the path to the mixed wav (ready to be muxed into the video).
    """
    tmp = work / "_dub"
    tmp.mkdir(parents=True, exist_ok=True)

    # 1. Extract + separate.
    print("  extracting audio…")
    raw_audio = extract_audio(video_path, tmp / "audio.wav")
    print("  separating voice from background (demucs)…")
    vocals, background = separate_vocals(
        raw_audio, tmp / "sep", model=cfg.demucs_model, device=cfg.device,
    )

    # 2. Normalise the background to a known format for numpy mixing.
    bg_norm = tmp / "background.wav"
    run_checked(
        ["ffmpeg", "-y", "-i", str(background),
         "-ac", str(DUB_CHANNELS), "-ar", str(DUB_SR),
         "-c:a", "pcm_s16le", str(bg_norm)],
        "background normalise",
    )

    # 3. Resolve / clone the voice (ElevenLabs only; OpenAI uses a preset voice).
    voice_id = cfg.voice
    created_voice: Optional[str] = None
    if cfg.engine == "elevenlabs":
        if cfg.clone_voice:
            print("  cloning the original speaker's voice (ElevenLabs IVC)…")
            sample = tmp / "voice_sample.wav"
            # ElevenLabs recommends ~1 min of clean speech; the separated vocals
            # are exactly that. Cap length/channels to keep the upload small.
            run_checked(
                ["ffmpeg", "-y", "-i", str(vocals), "-t", "120",
                 "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(sample)],
                "voice sample",
            )
            voice_id = elevenlabs_clone_voice(
                sample, cfg.api_key, name=f"youtranslate-{video_path.stem[:40]}",
            )
            created_voice = voice_id
            print(f"  cloned → voice {voice_id}")
        else:
            voice_id = elevenlabs_resolve_voice(cfg.voice, cfg.api_key)

    # 4. Synthesise + time-fit each segment.
    clips_dir = tmp / "clips"
    clips_dir.mkdir(exist_ok=True)
    lang_code = iso1(target_lang)
    placements: List[Tuple[float, Path]] = []
    total = len(tgt_segments)
    try:
        for i, seg in enumerate(tgt_segments, start=1):
            if not seg.text.strip():
                continue
            raw_clip = clips_dir / f"seg_{i:05d}.mp3"
            synthesize_segment(
                cfg, seg.text, raw_clip, voice_id=voice_id, lang_code=lang_code,
            )
            fitted = clips_dir / f"seg_{i:05d}.wav"
            slot = max(0.0, seg.end - seg.start)
            fit_clip(raw_clip, fitted, slot, max_tempo=cfg.max_tempo)
            placements.append((seg.start, fitted))
            print(f"  synthesized {i}/{total}", end="\r")
        print()
    finally:
        if created_voice and not cfg.keep_clone:
            elevenlabs_delete_voice(created_voice, cfg.api_key)

    if not placements:
        raise SystemExit("dubbing produced no speech clips; nothing to mix.")

    # 5. Assemble timeline + mix over background.
    print("  mixing dubbed speech over the background…")
    mixed = assemble_and_mix(
        bg_norm, placements, tmp / "mixed.wav",
        bg_gain=cfg.bg_gain, voice_gain=cfg.voice_gain,
    )
    return mixed


# --- Orchestration ----------------------------------------------------------


def compose_output_name(
    base: str, src: str, tgt: str, *, dub: bool, subtitled: bool,
) -> str:
    """Build the output filename: <base>.<src>.<tgt>[.dubbed][.subtitled].mp4."""
    name = f"{base}.{src}.{tgt}"
    if dub:
        name += ".dubbed"
    if subtitled:
        name += ".subtitled"
    return name + ".mp4"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="youtranslate",
        description=(
            "Download a YouTube video, transcribe it, translate the speech to "
            "another language, and either burn translated subtitles into the "
            "video or dub it (replace the voices, keep the background)."
        ),
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("./out"),
        help="Output directory (default: ./out)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER[+KEYRING][:PROFILE][::CONTAINER]",
        help=(
            "Load cookies from a browser for yt-dlp (for example: chrome). "
            "Useful when YouTube rejects an unauthenticated download with HTTP 403."
        ),
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small). tiny=fastest, large-v3=most accurate.",
    )
    parser.add_argument(
        "--engine",
        default="openai-whisper",
        choices=["faster-whisper", "openai-whisper"],
        help="Transcription engine (default: openai-whisper).",
    )
    parser.add_argument(
        "--source-lang",
        default="auto",
        help=(
            "Source language code (e.g. en, fr, de, ja) or 'auto' to let "
            "Whisper detect from the audio. Default: auto."
        ),
    )
    parser.add_argument(
        "--translator",
        default="openai",
        choices=["deep", "openai"],
        help="Translation backend (default: openai).",
    )
    parser.add_argument(
        "--target-lang",
        default="de",
        help=(
            "Target language code for the translation (default: de = German). "
            "Examples: fr, es, ja, pt-BR."
        ),
    )
    parser.add_argument(
        "--keep-srt",
        action="store_true",
        help=(
            "Keep the .<src>.srt and .<src>.<tgt>.srt subtitle files alongside "
            "the output video. By default only the video is kept."
        ),
    )

    # --- Subtitle output -----------------------------------------------------
    subs = parser.add_argument_group("subtitles")
    subs.add_argument(
        "--subtitles",
        default="burn",
        choices=["burn", "soft", "none"],
        help=(
            "Subtitle handling (default: burn). 'burn' hardcodes them into the "
            "video, 'soft' attaches a toggleable mov_text track, 'none' omits "
            "subtitles (use with --dub for an audio-only dub)."
        ),
    )
    subs.add_argument(
        "--soft-subs",
        action="store_true",
        help="Deprecated alias for --subtitles soft.",
    )
    subs.add_argument(
        "--font", default="Arial",
        help="Subtitle font face (default: Arial). Set to e.g. DejaVu Sans on Linux.",
    )
    subs.add_argument(
        "--font-size", type=int, default=None,
        help="Subtitle font size in pixels (auto by default based on resolution).",
    )
    subs.add_argument(
        "--margin-v", type=int, default=30,
        help="Vertical margin (in pixels) from the bottom of the frame (default: 30).",
    )
    subs.add_argument(
        "--burn-encoder",
        default="auto",
        choices=["auto", "nvenc", "libx264"],
        help=(
            "Video encoder when burning subtitles: 'auto' picks h264_nvenc when "
            "available, else libx264. Ignored unless --subtitles burn."
        ),
    )

    # --- Dubbing -------------------------------------------------------------
    dubg = parser.add_argument_group("dubbing")
    dubg.add_argument(
        "--dub",
        action="store_true",
        help=(
            "Replace the original voices with generated, translated speech while "
            "keeping the background audio (music/ambience/noise). Requires demucs "
            "and a TTS backend."
        ),
    )
    dubg.add_argument(
        "--tts",
        default="elevenlabs",
        choices=["elevenlabs", "openai"],
        help="Text-to-speech backend for --dub (default: elevenlabs).",
    )
    dubg.add_argument(
        "--tts-voice",
        default=None,
        help=(
            "Voice for the dub. ElevenLabs: a voice ID or a voice name "
            "(default: Rachel). OpenAI: a voice name like alloy/nova (default: alloy)."
        ),
    )
    dubg.add_argument(
        "--tts-model",
        default=None,
        help=(
            "TTS model id. ElevenLabs default: eleven_multilingual_v2. "
            "OpenAI default: gpt-4o-mini-tts."
        ),
    )
    dubg.add_argument(
        "--clone-voice",
        action="store_true",
        help=(
            "ElevenLabs only: clone the original speaker's voice from the "
            "separated vocals so the dub keeps their timbre. Requires a paid "
            "ElevenLabs plan. Only use on voices you're allowed to clone."
        ),
    )
    dubg.add_argument(
        "--keep-clone",
        action="store_true",
        help="Keep the cloned ElevenLabs voice in your account (default: delete it).",
    )
    dubg.add_argument(
        "--keep-dub-audio",
        action="store_true",
        help=(
            "Keep the dubbing work dir (separated stems, per-segment clips, "
            "mixed track). By default it is deleted after rendering."
        ),
    )
    dubg.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name for source separation (default: htdemucs).",
    )
    dubg.add_argument(
        "--dub-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Demucs separation (default: auto).",
    )
    dubg.add_argument(
        "--dub-max-tempo",
        type=float,
        default=1.5,
        help=(
            "Max speed-up applied to a synthesized clip to fit its slot "
            "(default: 1.5, clamped to 2.0). Higher keeps A/V sync tighter but "
            "can sound rushed."
        ),
    )
    dubg.add_argument(
        "--dub-bg-gain",
        type=float,
        default=0.7,
        help="Background loudness multiplier in the mix (default: 0.7).",
    )
    dubg.add_argument(
        "--dub-voice-gain",
        type=float,
        default=1.0,
        help="Dubbed-voice loudness multiplier in the mix (default: 1.0).",
    )
    args = parser.parse_args(argv)

    # Resolve subtitle mode (--soft-subs is a back-compat alias).
    subtitles_mode = "soft" if args.soft_subs else args.subtitles
    want_subs = subtitles_mode in ("burn", "soft")

    if not args.dub and not want_subs:
        raise SystemExit(
            "nothing to produce: --subtitles none needs --dub, otherwise choose "
            "--subtitles burn|soft."
        )
    if args.dub_max_tempo < 1.0 or args.dub_max_tempo > 2.0:
        raise SystemExit("--dub-max-tempo must be between 1.0 and 2.0.")

    # Fail fast on missing keys — before any download / transcription work.
    if args.translator == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "or pass --translator deep to use the free Google Translate backend."
        )
    dub_cfg: Optional[DubConfig] = None
    if args.dub:
        if args.tts == "elevenlabs":
            key = os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                raise SystemExit(
                    "ELEVENLABS_API_KEY is not set. Export it before running:\n"
                    "    export ELEVENLABS_API_KEY=...\n"
                    "or pass --tts openai to use the OpenAI speech API instead."
                )
            voice = args.tts_voice or DEFAULT_ELEVEN_VOICE
            model = args.tts_model or DEFAULT_ELEVEN_MODEL
        else:  # openai
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit(
                    "OPENAI_API_KEY is not set but --tts openai was requested.\n"
                    "    export OPENAI_API_KEY=sk-..."
                )
            if args.clone_voice:
                raise SystemExit("--clone-voice is only supported with --tts elevenlabs.")
            voice = args.tts_voice or DEFAULT_OPENAI_TTS_VOICE
            model = args.tts_model or DEFAULT_OPENAI_TTS_MODEL
        dub_cfg = DubConfig(
            engine=args.tts, api_key=key, voice=voice, model=model,
            clone_voice=args.clone_voice, keep_clone=args.keep_clone,
            demucs_model=args.demucs_model, device=args.dub_device,
            max_tempo=args.dub_max_tempo, bg_gain=args.dub_bg_gain,
            voice_gain=args.dub_voice_gain,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    work: Path = args.output

    # Dynamic step numbering (dub adds a stage).
    step_labels = ["Download", "Transcribe", "Translate"]
    if args.dub:
        step_labels.append("Dub")
    step_labels.append("Render")
    total_steps = len(step_labels)
    step_no = 0

    def step(label: str) -> str:
        nonlocal step_no
        step_no += 1
        return f"[{step_no}/{total_steps}] {label}:"

    # Step 1: download
    print(f"{step('Download')} {args.url}")
    video_path = download(args.url, work, args.cookies_from_browser)
    base = video_path.stem

    # Step 2: transcribe
    print(f"{step('Transcribe')} {args.engine} (model={args.model}, "
          f"source={args.source_lang})")
    src_segments, detected_lang = transcribe(
        video_path, args.model, args.engine, args.source_lang
    )
    if not src_segments:
        raise SystemExit("transcription produced no segments; nothing to translate.")
    # If the user pinned --source-lang, that wins; otherwise use what Whisper
    # detected (could be e.g. "en" or "fr" depending on the audio).
    source_lang = args.source_lang if args.source_lang != "auto" else detected_lang
    source_human = human_lang_name(source_lang)
    print(f"  got {len(src_segments)} {source_human} segments")

    # Step 3: translate
    target_human = human_lang_name(args.target_lang)
    print(f"{step('Translate')} {source_human} → {target_human} "
          f"via {args.translator}")
    tgt_segments = translate_segments(
        src_segments, args.translator,
        target_lang=args.target_lang, target_lang_human=target_human,
    )

    src_srt = work / f"{base}.{source_lang}.srt"
    tgt_srt = work / f"{base}.{source_lang}.{args.target_lang}.srt"
    write_srt(src_segments, src_srt)
    write_srt(tgt_segments, tgt_srt)
    print(f"  wrote subtitles: {src_srt.name}, {tgt_srt.name}")

    # Step 4 (optional): dub — replace voices, keep background.
    dubbed_audio: Optional[Path] = None
    if args.dub:
        assert dub_cfg is not None
        clone_note = " with voice cloning" if dub_cfg.clone_voice else ""
        print(f"{step('Dub')} synthesizing {target_human} speech via "
              f"{dub_cfg.engine}{clone_note}")
        dubbed_audio = dub_audio(
            video_path, tgt_segments,
            target_lang=args.target_lang, work=work, cfg=dub_cfg,
        )

    # Final step: render the output video.
    burn = subtitles_mode == "burn"
    out_video = work / compose_output_name(
        base, source_lang, args.target_lang, dub=args.dub, subtitled=want_subs,
    )
    action = []
    if args.dub:
        action.append("muxing dub")
    if want_subs:
        action.append("burning subtitles" if burn else "attaching subtitles")
    print(f"{step('Render')} {', '.join(action)}")

    if burn:
        # Burn re-encodes the video; fold the dubbed audio in during the same pass.
        burn_subtitles(
            video_path, tgt_srt, out_video,
            font=args.font, font_size=args.font_size, margin_v=args.margin_v,
            encoder=args.burn_encoder, audio_path=dubbed_audio,
        )
    else:
        # No burn → stream-copy the video and (optionally) replace audio / add
        # a soft subtitle track.
        attach_or_replace(
            video_path, out_video,
            audio_path=dubbed_audio,
            soft_srt=tgt_srt if subtitles_mode == "soft" else None,
            target_lang=args.target_lang,
        )
    print(f"  wrote: {out_video.name}")

    if not args.keep_srt:
        for f in (src_srt, tgt_srt):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    if args.dub and not args.keep_dub_audio:
        shutil.rmtree(work / "_dub", ignore_errors=True)

    print()
    print("Done.")
    if args.dub:
        print(f"  Dubbed video ({target_human})       : {out_video}")
    elif want_subs:
        print(f"  Video with {target_human} subtitles : {out_video}")
    if args.keep_srt:
        print(f"  {source_human} subtitles            : {src_srt}")
        print(f"  {target_human} subtitles            : {tgt_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
