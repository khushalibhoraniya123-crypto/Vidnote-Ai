"""
Speech-to-Text Transcription service for VidNote AI using faster-whisper.
Handles audio transcription, timestamped segment generation, and database synchronization.
"""

import os
import wave
import logging
import numpy as np
from django.conf import settings
from django.db import transaction
from ..models import Video, TranscriptSegment
from .media import process_video_audio_extraction

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading weights repeatedly
_CACHED_WHISPER_MODEL = None
_CACHED_MODEL_KEY = None

class TranscriptionError(Exception):
    """Base exception for transcription failures."""
    pass

class AudioFileNotFoundError(TranscriptionError):
    """Raised when the audio file for transcription is missing."""
    pass

class WhisperModelLoadError(TranscriptionError):
    """Raised when faster-whisper model cannot be loaded."""
    pass


def get_whisper_model(model_size: str = None, device: str = None, compute_type: str = None):
    """
    Get or initialize a cached faster-whisper WhisperModel instance.
    """
    global _CACHED_WHISPER_MODEL, _CACHED_MODEL_KEY

    model_size = model_size or getattr(settings, 'WHISPER_MODEL_SIZE', 'tiny')
    device = device or getattr(settings, 'WHISPER_DEVICE', 'cpu')
    compute_type = compute_type or getattr(settings, 'WHISPER_COMPUTE_TYPE', 'int8')

    model_key = f"{model_size}_{device}_{compute_type}"

    if _CACHED_WHISPER_MODEL is not None and _CACHED_MODEL_KEY == model_key:
        return _CACHED_WHISPER_MODEL

    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type
        )
        _CACHED_WHISPER_MODEL = model
        _CACHED_MODEL_KEY = model_key
        return model
    except Exception as e:
        raise WhisperModelLoadError(f"Failed to load Whisper model '{model_size}': {str(e)}")


def load_wav_audio_to_numpy(audio_path: str) -> np.ndarray:
    """
    Load a 16-bit PCM WAV file directly into a float32 numpy array.
    Extremely fast, robust, and independent of external audio decoders.
    """
    try:
        with wave.open(audio_path, 'rb') as wf:
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                # 16-bit signed integer PCM
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                # 32-bit signed integer PCM
                audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            elif sampwidth == 1:
                # 8-bit unsigned PCM
                audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            else:
                audio = np.frombuffer(frames, dtype=np.float32)

            # Convert multi-channel to mono by averaging
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)

            return audio
    except wave.Error:
        # Fallback to binary audio path if not a standard wave file
        return audio_path


def transcribe_audio_file(
    audio_path: str,
    model_size: str = None,
    language: str = None,
    beam_size: int = 5
) -> dict:
    """
    Transcribe a WAV audio file into timestamped speech segments using faster-whisper.

    Parameters:
    - audio_path: Absolute path to the WAV audio file.
    - model_size: 'tiny', 'base', 'small', 'medium', etc.
    - language: Optional language code (e.g. 'en'). Auto-detected if None.
    - beam_size: Beam search width (default 5).

    Returns:
    - Dict with:
        - 'segments': List of dicts [{'start': float, 'end': float, 'text': str}]
        - 'language': Detected language code
        - 'duration': Total transcribed audio duration
        - 'full_text': Combined text of all segments
    """
    if not audio_path or not os.path.exists(audio_path):
        raise AudioFileNotFoundError(f"Audio file does not exist at: '{audio_path}'")

    if os.path.getsize(audio_path) == 0:
        raise AudioFileNotFoundError(f"Audio file '{os.path.basename(audio_path)}' is empty (0 bytes).")

    # Load audio array
    audio_input = load_wav_audio_to_numpy(audio_path)
    model = get_whisper_model(model_size=model_size)

    try:
        segments_gen, info = model.transcribe(
            audio_input,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        segments = []
        full_text_parts = []

        for seg in segments_gen:
            clean_text = seg.text.strip()
            if clean_text:
                segments.append({
                    'start': round(float(seg.start), 2),
                    'end': round(float(seg.end), 2),
                    'text': clean_text
                })
                full_text_parts.append(clean_text)

        detected_lang = info.language if hasattr(info, 'language') and info.language else 'en'
        total_duration = info.duration if hasattr(info, 'duration') and info.duration else 0.0

        return {
            'segments': segments,
            'language': detected_lang,
            'duration': round(float(total_duration), 2),
            'full_text': " ".join(full_text_parts)
        }

    except Exception as e:
        raise TranscriptionError(f"Speech-to-text transcription failed: {str(e)}")


def process_video_transcription(video_instance: Video, model_size: str = None) -> list:
    """
    Full pipeline service to transcribe a video's audio and store timestamped segments.
    
    1. Ensures audio has been extracted (extracts via FFmpeg if not yet done).
    2. Runs faster-whisper transcription.
    3. Saves TranscriptSegment records in the database inside a transaction.
    4. Updates Video status, language, and duration.
    """
    # 1. Check/perform audio extraction if needed
    if not video_instance.has_audio or not os.path.exists(video_instance.audio_file.path):
        process_video_audio_extraction(video_instance)

    # 2. Update status to processing
    video_instance.status = Video.STATUS_PROCESSING
    video_instance.error_message = ""
    video_instance.save(update_fields=['status', 'error_message'])

    try:
        audio_path = video_instance.audio_file.path
        result = transcribe_audio_file(audio_path, model_size=model_size)

        segments_data = result.get('segments', [])

        # 3. Store transcript segments in database atomically
        with transaction.atomic():
            # Clear previous segments if re-transcribing
            video_instance.transcript_segments.all().delete()

            segment_objects = []
            for order_idx, seg in enumerate(segments_data):
                segment_objects.append(
                    TranscriptSegment(
                        video=video_instance,
                        start_time=seg['start'],
                        end_time=seg['end'],
                        text=seg['text'],
                        segment_order=order_idx
                    )
                )

            if segment_objects:
                TranscriptSegment.objects.bulk_create(segment_objects)

            # Update video metadata
            video_instance.language = result.get('language', 'en')
            if result.get('duration') and not video_instance.duration:
                video_instance.duration = result.get('duration')

            video_instance.status = Video.STATUS_COMPLETED
            video_instance.error_message = ""
            video_instance.save(update_fields=['language', 'duration', 'status', 'error_message'])

        return video_instance.transcript_segments.all()

    except Exception as exc:
        video_instance.status = Video.STATUS_FAILED
        video_instance.error_message = str(exc)
        video_instance.save(update_fields=['status', 'error_message'])
        raise
