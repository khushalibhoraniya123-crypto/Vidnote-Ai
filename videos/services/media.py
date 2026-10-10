"""
Media processing service for VidNote AI.
Handles FFmpeg audio extraction, media validation, and audio conversions.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from django.conf import settings
from django.core.files import File

# Custom Exceptions for clean error handling
class FFmpegError(Exception):
    """Base exception for FFmpeg operations."""
    pass

class FFmpegNotFoundError(FFmpegError):
    """Raised when FFmpeg binary is missing or cannot be located."""
    pass

class InvalidVideoError(FFmpegError):
    """Raised when the input video file is invalid, corrupt, or missing."""
    pass

class AudioExtractionError(FFmpegError):
    """Raised when audio extraction fails during FFmpeg execution."""
    pass


def get_ffmpeg_path() -> str:
    """
    Locate FFmpeg executable:
    1. Custom path configured in settings.FFMPEG_PATH / .env
    2. Bundled binary via imageio_ffmpeg
    3. System PATH via shutil.which('ffmpeg')
    """
    # 1. Check settings / .env
    custom_path = getattr(settings, 'FFMPEG_PATH', None)
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    # 2. Check imageio_ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.isfile(ffmpeg_exe):
            return ffmpeg_exe
    except (ImportError, Exception):
        pass

    # 3. Check system PATH
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        return system_ffmpeg

    raise FFmpegNotFoundError(
        "FFmpeg executable not found. Please configure FFMPEG_PATH in your .env or install imageio-ffmpeg."
    )


def extract_media_duration(file_path: str, ffmpeg_bin: str = None) -> float:
    """
    Extract media duration in seconds by parsing FFmpeg header output.
    """
    if not ffmpeg_bin:
        ffmpeg_bin = get_ffmpeg_path()

    cmd = [ffmpeg_bin, "-i", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='ignore')

    # Parse 'Duration: 00:01:23.45' from stderr
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", result.stderr)
    if match:
        hours, minutes, seconds = match.groups()
        return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 2)
    return 0.0


def extract_audio_from_video(
    video_path: str,
    output_audio_path: str = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> dict:
    """
    Extract speech-optimized WAV audio from a video file using FFmpeg.

    Parameters:
    - video_path: Full path to input video file on disk.
    - output_audio_path: Optional destination path for the extracted WAV audio.
    - sample_rate: Audio sampling rate in Hz (default 16000 Hz, optimal for Whisper).
    - channels: Number of audio channels (default 1 = mono).

    Returns:
    - Dict with 'audio_path', 'duration', and 'file_size'.
    """
    # 1. Validate video path existence
    if not video_path or not os.path.exists(video_path):
        raise InvalidVideoError(f"Video file does not exist at path: '{video_path}'")

    if os.path.getsize(video_path) == 0:
        raise InvalidVideoError(f"The video file '{os.path.basename(video_path)}' is empty (0 bytes).")

    # 2. Determine FFmpeg binary
    ffmpeg_bin = get_ffmpeg_path()

    # 3. Set destination output path if not provided
    if not output_audio_path:
        video_p = Path(video_path)
        output_dir = video_p.parent
        output_audio_path = str(output_dir / f"{video_p.stem}_extracted.wav")
    else:
        out_dir = os.path.dirname(output_audio_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)


    # 4. Construct FFmpeg command
    # -y : Overwrite output file without confirmation
    # -i : Input video file
    # -vn : Disable video recording
    # -acodec pcm_s16le : Standard 16-bit uncompressed PCM WAV
    # -ar : Sampling frequency (16000 Hz for Whisper / Speech Recognition)
    # -ac : Audio channels (1 = mono)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_audio_path)
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='ignore',
            timeout=180
        )
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("Audio extraction timed out after 180 seconds.")
    except Exception as e:
        raise AudioExtractionError(f"Failed to execute FFmpeg process: {str(e)}")

    if process.returncode != 0:
        error_output = process.stderr.strip()
        if "does not contain any stream" in error_output or "Output file is empty" in error_output:
            raise AudioExtractionError("The video file does not contain an audio stream or the stream is unreadable.")
        if "Invalid data found when processing input" in error_output:
            raise InvalidVideoError("Invalid or corrupt video data found during processing.")
        raise AudioExtractionError(f"FFmpeg extraction failed (code {process.returncode}): {error_output[-300:]}")

    if not os.path.exists(output_audio_path) or os.path.getsize(output_audio_path) == 0:
        raise AudioExtractionError("Audio file extraction completed but output file is missing or 0 bytes.")

    duration = extract_media_duration(output_audio_path, ffmpeg_bin)

    return {
        'audio_path': output_audio_path,
        'duration': duration,
        'file_size': os.path.getsize(output_audio_path),
    }


def process_video_audio_extraction(video_instance) -> bool:
    """
    Service function to handle the full extraction lifecycle for a Video model instance.
    Updates status, extracts audio, saves the audio file field, and logs errors.
    """
    video_instance.status = video_instance.STATUS_PROCESSING
    video_instance.error_message = ""
    video_instance.save(update_fields=['status', 'error_message'])

    try:
        video_path = video_instance.video_file.path
        
        # Determine audio target directory inside MEDIA_ROOT
        media_root = Path(settings.MEDIA_ROOT)
        relative_audio_dir = Path("audio") / video_instance.created_at.strftime("%Y/%m")
        abs_audio_dir = media_root / relative_audio_dir
        abs_audio_dir.mkdir(parents=True, exist_ok=True)

        video_stem = Path(video_path).stem
        dest_wav_path = str(abs_audio_dir / f"{video_stem}_audio.wav")

        # Run audio extraction
        result = extract_audio_from_video(
            video_path=video_path,
            output_audio_path=dest_wav_path,
            sample_rate=16000,
            channels=1
        )

        # Save audio file to model instance
        rel_audio_path = str(relative_audio_dir / f"{video_stem}_audio.wav").replace("\\", "/")
        video_instance.audio_file.name = rel_audio_path
        video_instance.duration = result.get('duration')
        video_instance.status = video_instance.STATUS_COMPLETED
        video_instance.error_message = ""
        video_instance.save(update_fields=['audio_file', 'duration', 'status', 'error_message'])
        return True

    except Exception as exc:
        video_instance.status = video_instance.STATUS_FAILED
        video_instance.error_message = str(exc)
        video_instance.save(update_fields=['status', 'error_message'])
        raise
