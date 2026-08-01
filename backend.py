"""
AI Shorts Maker — backend
Turns a landscape video into vertical (9:16) shorts with animated,
word-by-word burned-in captions (CapCut style).

Two modes:
  * make_viral_shorts()  — "Auto" mode. Transcribes the whole video once,
    scores candidate segments with a heuristic "hookiness" score, and
    renders the best non-overlapping ones as separate shorts.
  * make_my_short()      — manual mode. You pick the exact start/end time
    of a single clip yourself.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from typing import BinaryIO

import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shorts_maker")

# ---------------------------------------------------------------------------
# moviepy v1 / v2 compatibility shim
# ---------------------------------------------------------------------------
# moviepy 2.0 removed `moviepy.editor` and renamed several clip methods
# (subclip -> subclipped, set_start -> with_start, .crop() -> vfx.Crop,
# fl_image -> image_transform). This shim lets the rest of the file use one
# consistent API regardless of which major version is installed.
try:
    from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip

    _MOVIEPY_V2 = False
except ImportError:  # moviepy >= 2.0
    from moviepy import CompositeVideoClip, ImageClip, VideoFileClip
    from moviepy import vfx

    _MOVIEPY_V2 = True


def _subclip(clip, start, end):
    return clip.subclipped(start, end) if _MOVIEPY_V2 else clip.subclip(start, end)


def _crop(clip, x_center, width, height):
    if _MOVIEPY_V2:
        return clip.with_effects([vfx.Crop(x_center=x_center, width=width, height=height)])
    return clip.crop(x_center=x_center, width=width, height=height)


def _with_start(clip, start):
    return clip.with_start(start) if _MOVIEPY_V2 else clip.set_start(start)


def _with_duration(clip, duration):
    return clip.with_duration(duration) if _MOVIEPY_V2 else clip.set_duration(duration)


def _with_position(clip, position):
    return clip.with_position(position) if _MOVIEPY_V2 else clip.set_position(position)


def _image_transform(clip, func):
    return clip.image_transform(func) if _MOVIEPY_V2 else clip.fl_image(func)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FONT_CANDIDATES = [
    "impact.ttf",
    "Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Impact.ttf",
    "C:\\Windows\\Fonts\\impact.ttf",
]

# Heuristic "hook" words used to score how likely a spoken moment is to grab
# attention. This is a cheap, offline stand-in for a real virality model —
# not a claim of true semantic understanding.
HOOK_KEYWORDS = [
    "secret", "never", "always", "shocking", "amazing", "insane", "crazy",
    "wow", "unbelievable", "warning", "mistake", "truth", "proof", "hack",
    "trick", "stop", "wait", "actually", "honestly", "literally", "worst",
    "best", "biggest", "huge", "free", "why", "how", "you won't believe",
    "no one tells you", "nobody talks about",
]

# (min_seconds, max_seconds) duration window to search within for each preset
LENGTH_PRESETS: dict[str, tuple[float, float]] = {
    "Auto": (15.0, 60.0),
    "15s": (12.0, 18.0),
    "30s": (25.0, 35.0),
    "45s": (40.0, 50.0),
    "60s": (55.0, 65.0),
}

NUM_SHORTS_OPTIONS = ["Auto", "1", "2", "3", "4", "5"]


@dataclass(frozen=True)
class ShortConfig:
    """Parameters for a single, manually-timed short (manual mode)."""

    start_sec: float
    end_sec: float
    output_path: str = "viral_short_output.mp4"
    whisper_model_size: str = "base"
    font_size: int = 55
    caption_band_height: int = 150
    caption_y_ratio: float = 0.7
    text_color: str = "yellow"
    stroke_color: str = "black"
    stroke_width: int = 4
    fps: int = 24

    def __post_init__(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")


@dataclass(frozen=True)
class RenderStyle:
    """Shared caption/render styling, used by auto mode."""

    whisper_model_size: str = "base"
    font_size: int = 55
    caption_band_height: int = 150
    caption_y_ratio: float = 0.7
    text_color: str = "yellow"
    stroke_color: str = "black"
    stroke_width: int = 4
    fps: int = 24
    auto_enhance: bool = True


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Window:
    start: float
    end: float
    text: str
    words: list[Word]
    score: float


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str) -> WhisperModel:
    logger.info("Loading Whisper model '%s'...", model_size)
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    logger.warning("No TrueType font found; falling back to Pillow's default bitmap font.")
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _save_upload_to_temp(uploaded_file: BinaryIO, suffix: str = ".mp4") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _crop_to_vertical(clip):
    width, height = clip.size
    target_width = int(height * 9 / 16)
    if target_width > width:
        return clip  # already narrower than 9:16 — nothing sensible to crop
    return _crop(clip, x_center=width / 2, width=target_width, height=height)


def _enhance_frame(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(frame)
    img = ImageEnhance.Contrast(img).enhance(1.12)
    img = ImageEnhance.Color(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return np.array(img)


def _apply_auto_enhance(clip):
    return _image_transform(clip, _enhance_frame)


def _render_word_image(word_text: str, frame_width: int, font_size: int, band_height: int,
                        text_color: str, stroke_color: str, stroke_width: int) -> np.ndarray:
    font = _load_font(font_size)
    band = Image.new("RGBA", (frame_width, band_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)

    left, top, right, bottom = draw.textbbox((0, 0), word_text, font=font)
    text_w, text_h = right - left, bottom - top
    x = (frame_width - text_w) // 2
    y = (band_height - text_h) // 2

    draw.text((x, y), word_text, font=font, fill=text_color,
               stroke_width=stroke_width, stroke_fill=stroke_color)
    return np.array(band)


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def _build_caption_clips_manual(segments, config: ShortConfig, clip_duration: float,
                                 frame_width: int, frame_height: int):
    """
    Build caption clips from a transcript of an already-trimmed clip.
    Word timestamps here are relative to the trimmed clip (0..clip_duration),
    NOT the original video, so they must not be re-offset by start_sec.
    """
    caption_clips = []
    for segment in segments:
        for word in segment.words:
            if word.start < 0 or word.end > clip_duration:
                continue
            clean_word = word.word.strip().upper()
            if not clean_word:
                continue
            duration = max(word.end - word.start, 0.1)
            frame = _render_word_image(
                clean_word, frame_width, config.font_size, config.caption_band_height,
                config.text_color, config.stroke_color, config.stroke_width,
            )
            word_clip = ImageClip(frame, ismask=False)
            word_clip = _with_start(word_clip, word.start)
            word_clip = _with_duration(word_clip, duration)
            word_clip = _with_position(word_clip, ("center", frame_height * config.caption_y_ratio))
            caption_clips.append(word_clip)
    return caption_clips


def make_my_short(video_file: BinaryIO, start_sec: float, end_sec: float,
                   config: ShortConfig | None = None) -> str:
    """Manual mode: render exactly one short from a user-picked time range."""
    config = config or ShortConfig(start_sec=start_sec, end_sec=end_sec)

    temp_input_path = _save_upload_to_temp(video_file)
    audio_path = None
    source_clip = None
    vertical_clip = None
    final_video = None

    try:
        logger.info("Loading and trimming video...")
        source_clip = _subclip(VideoFileClip(temp_input_path), config.start_sec, config.end_sec)
        vertical_clip = _crop_to_vertical(source_clip)
        clip_duration = config.end_sec - config.start_sec

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            audio_path = tmp_audio.name
        source_clip.audio.write_audiofile(
            audio_path, fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
        )

        logger.info("Transcribing speech...")
        model = _load_whisper_model(config.whisper_model_size)
        segments, _info = model.transcribe(audio_path, word_timestamps=True)

        logger.info("Building animated captions...")
        frame_width, frame_height = vertical_clip.size
        caption_clips = _build_caption_clips_manual(
            list(segments), config, clip_duration, frame_width, frame_height
        )

        logger.info("Rendering final video (%d caption frames)...", len(caption_clips))
        final_video = CompositeVideoClip([vertical_clip] + caption_clips)
        final_video.write_videofile(
            config.output_path, codec="libx264", audio_codec="aac", fps=config.fps, logger=None
        )

        logger.info("Done: %s", config.output_path)
        return config.output_path

    finally:
        for obj in (final_video, vertical_clip, source_clip):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        for path in (audio_path, temp_input_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Could not remove temp file %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Auto mode — highlight detection + multi-short rendering
# ---------------------------------------------------------------------------

def _flatten_segments(raw_segments) -> list[Segment]:
    segments = []
    for seg in raw_segments:
        words = [Word(start=w.start, end=w.end, text=w.word) for w in (seg.words or [])]
        segments.append(Segment(start=seg.start, end=seg.end, text=seg.text or "", words=words))
    return segments


def _score_window(text: str, word_count: int, duration: float) -> float:
    """
    Cheap heuristic "hookiness" score — rewards punctuation that signals
    excitement/questions, hook keywords, numbers/stats, and brisk pacing.
    This is NOT a semantic understanding of the content, just a fast,
    fully-offline proxy for "this moment sounds engaging".
    """
    if duration <= 0:
        return 0.0
    lower = text.lower()
    score = 0.0
    score += lower.count("!") * 2.0
    score += lower.count("?") * 1.5
    score += sum(lower.count(k) for k in HOOK_KEYWORDS) * 3.0
    score += len(re.findall(r"\d+", text)) * 1.5
    pace = word_count / duration
    score += min(pace, 4.0)  # reward brisk delivery, but cap the reward
    return score


def _build_candidate_windows(segments: list[Segment], min_len: float, max_len: float) -> list[Window]:
    """
    Slide over whisper's natural (pause-based) segment boundaries and build
    candidate windows whose duration falls inside [min_len, max_len].
    """
    windows: list[Window] = []
    n = len(segments)
    for i in range(n):
        acc_words: list[Word] = []
        acc_text: list[str] = []
        start = segments[i].start
        for j in range(i, n):
            acc_text.append(segments[j].text)
            acc_words.extend(segments[j].words)
            end = segments[j].end
            duration = end - start
            if duration < min_len:
                continue
            if duration > max_len:
                break
            text = " ".join(t.strip() for t in acc_text if t.strip())
            score = _score_window(text, len(acc_words), duration)
            windows.append(Window(start=start, end=end, text=text, words=list(acc_words), score=score))
    return windows


def _words_in_range(segments: list[Segment], start: float, end: float) -> list[Word]:
    return [w for seg in segments for w in seg.words if w.start >= start and w.end <= end]


def _select_windows(windows: list[Window], target_count: int | None) -> list[Window]:
    """Greedily pick the highest-scoring windows that don't overlap each other."""
    windows_sorted = sorted(windows, key=lambda w: w.score, reverse=True)
    selected: list[Window] = []
    cap = target_count if target_count is not None else 6
    for w in windows_sorted:
        if len(selected) >= cap:
            break
        overlaps = any(not (w.end <= s.start or w.start >= s.end) for s in selected)
        if overlaps:
            continue
        selected.append(w)
    selected.sort(key=lambda w: w.start)
    return selected


def _build_caption_clips_from_words(words: list[Word], window_start: float, duration: float,
                                     style: RenderStyle, frame_width: int, frame_height: int):
    clips = []
    for word in words:
        rel_start = word.start - window_start
        rel_end = word.end - window_start
        if rel_end <= 0 or rel_start >= duration:
            continue
        rel_start = max(rel_start, 0.0)
        clean = word.text.strip().upper()
        if not clean:
            continue
        dur = max(rel_end - rel_start, 0.1)
        frame = _render_word_image(
            clean, frame_width, style.font_size, style.caption_band_height,
            style.text_color, style.stroke_color, style.stroke_width,
        )
        wc = ImageClip(frame, ismask=False)
        wc = _with_start(wc, rel_start)
        wc = _with_duration(wc, dur)
        wc = _with_position(wc, ("center", frame_height * style.caption_y_ratio))
        clips.append(wc)
    return clips


def _render_short(source_clip, window: Window, style: RenderStyle, output_path: str) -> str:
    clip = _subclip(source_clip, window.start, window.end)
    vertical = _crop_to_vertical(clip)
    if style.auto_enhance:
        vertical = _apply_auto_enhance(vertical)

    frame_w, frame_h = vertical.size
    duration = window.end - window.start
    caption_clips = _build_caption_clips_from_words(window.words, window.start, duration, style, frame_w, frame_h)

    final = CompositeVideoClip([vertical] + caption_clips)
    try:
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=style.fps, logger=None)
    finally:
        for c in (final, vertical, clip):
            try:
                c.close()
            except Exception:
                pass
    return output_path


def make_viral_shorts(video_file: BinaryIO, num_shorts_choice: str, length_choice: str,
                       auto_enhance: bool, whisper_model_size: str = "base", font_size: int = 55,
                       output_dir: str = ".") -> list[str]:
    """
    Auto mode: transcribes the whole video once, finds the best non-
    overlapping "hooky" moments, and renders each as its own vertical short.

    num_shorts_choice: "Auto" or a string integer ("1".."5")
    length_choice: one of LENGTH_PRESETS keys ("Auto", "15s", "30s", "45s", "60s")
    """
    style = RenderStyle(whisper_model_size=whisper_model_size, font_size=font_size, auto_enhance=auto_enhance)
    min_len, max_len = LENGTH_PRESETS.get(length_choice, LENGTH_PRESETS["Auto"])

    temp_input_path = _save_upload_to_temp(video_file)
    audio_path = None
    source_clip = None
    output_paths: list[str] = []

    try:
        logger.info("Loading video...")
        source_clip = VideoFileClip(temp_input_path)
        video_duration = source_clip.duration

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            audio_path = tmp_audio.name
        source_clip.audio.write_audiofile(
            audio_path, fps=16000, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
        )

        logger.info("Transcribing full video once...")
        model = _load_whisper_model(style.whisper_model_size)
        raw_segments, _info = model.transcribe(audio_path, word_timestamps=True)
        segments = _flatten_segments(raw_segments)

        if not segments:
            raise RuntimeError("Video mein koi speech detect nahi hui — highlights nahi ban sakte.")

        effective_max = min(max_len, video_duration)
        windows = _build_candidate_windows(segments, min_len, effective_max)

        if not windows:
            logger.warning("No window matched the length preset; falling back to the whole clip.")
            fallback_end = min(video_duration, max_len)
            windows = [Window(
                start=0.0, end=fallback_end, text="",
                words=_words_in_range(segments, 0.0, fallback_end), score=0.0,
            )]

        target_count = None if num_shorts_choice == "Auto" else int(num_shorts_choice)
        if target_count is None:
            target_count = max(1, min(6, round(video_duration / 90)))

        selected = _select_windows(windows, target_count)

        logger.info("Rendering %d short(s)...", len(selected))
        os.makedirs(output_dir, exist_ok=True)
        for idx, window in enumerate(selected, start=1):
            output_path = os.path.join(output_dir, f"viral_short_{idx}.mp4")
            _render_short(source_clip, window, style, output_path)
            output_paths.append(output_path)
            logger.info("Rendered short %d/%d: %.1fs-%.1fs (score %.1f)",
                        idx, len(selected), window.start, window.end, window.score)

        return output_paths

    finally:
        if source_clip is not None:
            try:
                source_clip.close()
            except Exception:
                pass
        for path in (audio_path, temp_input_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Could not remove temp file %s: %s", path, exc)