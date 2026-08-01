"""
AI Shorts Maker — backend
Converts a landscape video into a 9:16 vertical short with animated,
word-by-word burned-in captions (CapCut style).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import BinaryIO

import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# moviepy v1 / v2 compatibility shim
# ---------------------------------------------------------------------------
# moviepy 2.0 removed the `moviepy.editor` module and renamed several clip
# methods (subclip -> subclipped, set_start -> with_start, etc). This shim
# lets the rest of the file use one consistent API regardless of which
# major version is installed.
try:
    from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip

    _MOVIEPY_V2 = False
except ImportError:  # moviepy >= 2.0
    from moviepy import CompositeVideoClip, ImageClip, VideoFileClip
    from moviepy import vfx

    _MOVIEPY_V2 = True


def _subclip(clip, start, end):
    if _MOVIEPY_V2:
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("shorts_maker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fonts to try, in order, on top of a hard-coded fallback that always exists
# alongside Pillow. This keeps the app working on Linux/servers where
# "impact.ttf" is not installed (the original code silently fell back to a
# tiny default bitmap font in that case).
FONT_CANDIDATES = [
    "impact.ttf",
    "Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Impact.ttf",
    "C:\\Windows\\Fonts\\impact.ttf",
]


@dataclass(frozen=True)
class ShortConfig:
    """Tunable parameters for a single short generation job."""

    start_sec: float
    end_sec: float
    output_path: str = "viral_short_output.mp4"
    whisper_model_size: str = "base"
    font_size: int = 55
    caption_band_height: int = 150
    caption_y_ratio: float = 0.7  # vertical position of captions, 0=top 1=bottom
    text_color: str = "yellow"
    stroke_color: str = "black"
    stroke_width: int = 4
    fps: int = 24

    def __post_init__(self):
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")


# ---------------------------------------------------------------------------
# Cached resources (loading the Whisper model is expensive — do it once)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str) -> WhisperModel:
    logger.info("Loading Whisper model '%s'...", model_size)
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    logger.warning("No TrueType font found; falling back to Pillow's default bitmap font.")
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload_to_temp(uploaded_file: BinaryIO, suffix: str = ".mp4") -> str:
    """Persist an in-memory upload (e.g. a Streamlit UploadedFile) to disk."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _crop_to_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Center-crop a landscape clip down to a 9:16 vertical frame."""
    width, height = clip.size
    target_width = int(height * 9 / 16)
    if target_width > width:
        # Source is already narrower than 9:16 — nothing sensible to crop.
        return clip
    return _crop(clip, x_center=width / 2, width=target_width, height=height)


def _render_word_image(word_text: str, config: ShortConfig, frame_width: int) -> np.ndarray:
    """Render a single caption word as an RGBA numpy array, centered."""
    font = _load_font(config.font_size)
    band = Image.new("RGBA", (frame_width, config.caption_band_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)

    left, top, right, bottom = draw.textbbox((0, 0), word_text, font=font)
    text_w, text_h = right - left, bottom - top
    x = (frame_width - text_w) // 2
    y = (config.caption_band_height - text_h) // 2

    draw.text(
        (x, y),
        word_text,
        font=font,
        fill=config.text_color,
        stroke_width=config.stroke_width,
        stroke_fill=config.stroke_color,
    )
    return np.array(band)


def _build_caption_clips(segments, config: ShortConfig, clip_duration: float, frame_width: int, frame_height: int):
    """
    Build one ImageClip per transcribed word, positioned to appear only
    while that word is being spoken.

    Note: `segments` comes from transcribing audio that was already trimmed
    to [start_sec, end_sec], so word.start / word.end are relative to the
    trimmed clip (i.e. already in the 0..clip_duration range) — they must
    NOT be re-offset by start_sec.
    """
    caption_clips = []
    for segment in segments:
        for word in segment.words:
            if word.start < 0 or word.end > clip_duration:
                continue  # word timestamp falls outside the trimmed clip

            clean_word = word.word.strip().upper()
            if not clean_word:
                continue

            duration = max(word.end - word.start, 0.1)
            frame = _render_word_image(clean_word, config, frame_width)

            word_clip = ImageClip(frame, ismask=False)
            word_clip = _with_start(word_clip, word.start)
            word_clip = _with_duration(word_clip, duration)
            word_clip = _with_position(word_clip, ("center", frame_height * config.caption_y_ratio))
            caption_clips.append(word_clip)
    return caption_clips


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_my_short(video_file: BinaryIO, start_sec: float, end_sec: float,
                   config: ShortConfig | None = None) -> str:
    """
    Turn an uploaded landscape video into a vertical (9:16) short with
    animated word-by-word captions burned in.

    Returns the path to the rendered .mp4 file.
    """
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
        caption_clips = _build_caption_clips(
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
        # Release moviepy/ffmpeg file handles before deleting temp files,
        # otherwise cleanup can fail (especially on Windows).
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