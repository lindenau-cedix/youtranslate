#!/usr/bin/env python3
"""
youtranslate — Download a YouTube video in any language, transcribe it,
translate the speech to another language, and burn the translated subtitles
into the video.

Pipeline:
    1. yt-dlp downloads the video (best video + best audio, muxed to mp4).
    2. faster-whisper (or openai-whisper) transcribes the audio to segments
       in the source language.
    3. deep-translator (Google, free) or OpenAI translates each segment to the
       target language.
    4. ffmpeg + libass burns the translated SRT into a copy of the video.

Outputs (in --output dir):
    <title>.<src>.<tgt>.subtitled.mp4   — video with translated subtitles burned in.
    <title>.<src>.srt                  — original source subtitles (kept for reference).
    <title>.<src>.<tgt>.srt            — translated subtitles (kept for reference unless
                                         --no-keep-srt is passed).

Example:
    python youtranslate.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \\
        --output ./out --model small --source-lang en --target-lang de
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
from typing import List, Optional, Sequence


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


# --- Step 1: download -------------------------------------------------------


def download(url: str, out_dir: Path) -> Path:
    """Download the given YouTube URL into out_dir; return the resulting file path."""
    ensure_tool("yt-dlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--merge-output-format", "mp4",
        "-f", "bv*+ba/b",
        "-o", str(out_dir / "%(title)s.%(ext)s"),
        url,
    ]
    print("  $ yt-dlp " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # yt-dlp prints useful diagnostics to stderr when it fails.
        sys.stderr.write(proc.stderr or proc.stdout or "")
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


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    out_path: Path,
    *,
    font: str = "Arial",
    font_size: Optional[int] = None,
    margin_v: int = 30,
    encoder: str = "auto",
) -> None:
    """Burn SRT subtitles into a copy of the video using ffmpeg + libass.

    `encoder` selects the video encoder:
      - "auto"    : use h264_nvenc if ffmpeg was built with NVENC, else libx264.
      - "nvenc"   : force h264_nvenc (errors out if ffmpeg doesn't support it).
      - "libx264" : force the CPU encoder.
    """
    ensure_tool("ffmpeg")
    ensure_tool("ffprobe")

    width, height = ffprobe_dimensions(video_path)
    # Auto-pick a sensible font size based on resolution.
    if font_size is None:
        font_size = max(16, min(48, int(height / 28)))

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

    # Resolve the encoder. NVENC gives 5–15× speedup over libx264 with no
    # perceptual quality loss; libx264 is the universal fallback.
    if encoder == "auto":
        use_nvenc = ffmpeg_has_encoder("h264_nvenc")
        if use_nvenc:
            print("  using NVIDIA NVENC encoder (h264_nvenc)")
        else:
            print("  using CPU encoder (libx264) — "
                  "h264_nvenc not found in this ffmpeg build")
    elif encoder == "nvenc":
        if not ffmpeg_has_encoder("h264_nvenc"):
            raise SystemExit(
                "--burn-encoder nvenc requested, but this ffmpeg build does not "
                "advertise h264_nvenc. See README for instructions on building "
                "ffmpeg with --enable-nvenc."
            )
        use_nvenc = True
        print("  using NVIDIA NVENC encoder (h264_nvenc)")
    elif encoder == "libx264":
        use_nvenc = False
        print("  using CPU encoder (libx264)")
    else:
        raise SystemExit(f"unknown --burn-encoder '{encoder}'")

    if use_nvenc:
        # NVENC: `-rc vbr -cq N -b:v 0` is the constant-quality mode analogous
        # to libx264's CRF. `-preset p4` ≈ libx264's "medium". `-hwaccel cuda`
        # decodes on the GPU too, shaving more time off large videos.
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel", "cuda",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "22",
            "-b:v", "0",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
    print("  $ ffmpeg " + " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout or "")
        raise SystemExit(f"ffmpeg failed with exit code {proc.returncode}")


# --- Orchestration ----------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="youtranslate",
        description=(
            "Download a YouTube video, transcribe it, translate the speech to "
            "another language, and burn the translated subtitles into the video."
        ),
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("./out"),
        help="Output directory (default: ./out)",
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
            "Target language code for the subtitles (default: de = German). "
            "Examples: fr, es, ja, pt-BR."
        ),
    )
    parser.add_argument(
        "--keep-srt",
        action="store_true",
        help=(
            "Keep the .<src>.srt and .<src>.<tgt>.srt subtitle files alongside "
            "the subtitled video. By default only the subtitled video is kept."
        ),
    )
    parser.add_argument(
        "--font", default="Arial",
        help="Subtitle font face (default: Arial). Set to e.g. DejaVu Sans on Linux.",
    )
    parser.add_argument(
        "--font-size", type=int, default=None,
        help="Subtitle font size in pixels (auto by default based on resolution).",
    )
    parser.add_argument(
        "--margin-v", type=int, default=30,
        help="Vertical margin (in pixels) from the bottom of the frame (default: 30).",
    )
    parser.add_argument(
        "--soft-subs",
        action="store_true",
        help=(
            "Instead of burning the subtitles into the video, attach them as a "
            "mov_text sidecar stream in the .<src>.<tgt>.mp4 container."
        ),
    )
    parser.add_argument(
        "--burn-encoder",
        default="auto",
        choices=["auto", "nvenc", "libx264"],
        help=(
            "Video encoder for the subtitled output. 'auto' picks h264_nvenc "
            "when available, else libx264. 'nvenc' requires an ffmpeg build with "
            "--enable-nvenc and an NVIDIA GPU. Ignored with --soft-subs."
        ),
    )
    args = parser.parse_args(argv)

    # Fail fast with a clear message if the user picked an OpenAI backend but
    # didn't set the key — without this we wait until the first translation call.
    if args.translator == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it before running:\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "or pass --translator deep to use the free Google Translate backend."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    work: Path = args.output

    # Step 1: download
    print(f"[1/4] Downloading {args.url}")
    video_path = download(args.url, work)
    base = video_path.stem

    # Step 2: transcribe
    print(f"[2/4] Transcribing with {args.engine} (model={args.model}, "
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
    print(f"[3/4] Translating from {source_human} → {target_human} "
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

    # Step 4: burn (or attach) subtitles
    print(f"[4/4] {'Burning' if not args.soft_subs else 'Attaching'} subtitles")
    out_video = work / f"{base}.{source_lang}.{args.target_lang}.subtitled.mp4"

    if args.soft_subs:
        # Attach as a separate (mov_text) stream — viewer can toggle them on/off.
        ensure_tool("ffmpeg")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(tgt_srt),
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", f"language={args.target_lang}",
            "-movflags", "+faststart",
            str(out_video),
        ]
        print("  $ ffmpeg " + " ".join(cmd[1:]))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or proc.stdout or "")
            raise SystemExit(f"ffmpeg failed with exit code {proc.returncode}")
    else:
        burn_subtitles(
            video_path,
            tgt_srt,
            out_video,
            font=args.font,
            font_size=args.font_size,
            margin_v=args.margin_v,
            encoder=args.burn_encoder,
        )
    print(f"  wrote: {out_video.name}")

    if not args.keep_srt:
        for f in (src_srt, tgt_srt):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    print()
    print("Done.")
    print(f"  Video with {target_human} subtitles : {out_video}")
    if args.keep_srt:
        print(f"  {source_human} subtitles            : {src_srt}")
        print(f"  {target_human} subtitles            : {tgt_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
