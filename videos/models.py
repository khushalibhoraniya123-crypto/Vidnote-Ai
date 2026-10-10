import os
from django.db import models
from django.contrib.auth.models import User

def format_seconds_to_hhmmss(seconds_val: float) -> str:
    """
    Format a floating point or integer duration into HH:MM:SS timestamp format.
    """
    if seconds_val is None or seconds_val < 0:
        return "00:00:00"
    total_seconds = int(seconds_val)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Video(models.Model):
    """
    Model representing an uploaded video for processing, summarization, and RAG.
    """
    STATUS_UPLOADED = 'uploaded'
    STATUS_QUEUED = 'queued'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_UPLOADED, 'Uploaded'),
        (STATUS_QUEUED, 'Queued'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='videos',
        help_text="The user who uploaded this video"
    )
    title = models.CharField(max_length=255, help_text="Video title or lecture topic")
    description = models.TextField(blank=True, help_text="Optional description or notes about this video")
    video_file = models.FileField(
        upload_to='videos/%Y/%m/',
        help_text="Uploaded video file (MP4, MKV, MOV, AVI, WEBM)"
    )
    original_filename = models.CharField(max_length=255, blank=True)
    file_size = models.PositiveBigIntegerField(default=0, help_text="File size in bytes")
    duration = models.FloatField(null=True, blank=True, help_text="Duration in seconds")
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_UPLOADED,
        db_index=True
    )
    audio_file = models.FileField(
        upload_to='audio/%Y/%m/',
        null=True,
        blank=True,
        help_text="Extracted audio file (WAV format for speech recognition)"
    )
    language = models.CharField(
        max_length=20,
        blank=True,
        default='en',
        help_text="Detected or specified speech language (e.g. 'en')"
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error message if audio extraction, transcription, or summarization failed"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Video'
        verbose_name_plural = 'Videos'

    def __str__(self):
        return f"{self.title} ({self.user.username}) - {self.get_status_display()}"

    def save(self, *args, **kwargs):
        if self.video_file and not self.original_filename:
            self.original_filename = os.path.basename(self.video_file.name)
        if self.video_file and (self.file_size == 0 or self.file_size is None):
            try:
                self.file_size = self.video_file.size
            except (AttributeError, FileNotFoundError, ValueError):
                pass
        super().save(*args, **kwargs)

    @property
    def formatted_file_size(self):
        if not self.file_size:
            return "0 KB"
        size = float(self.file_size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0 or unit == 'TB':
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} MB"

    @property
    def formatted_duration(self):
        if not self.duration:
            return "00:00:00"
        return format_seconds_to_hhmmss(self.duration)

    @property
    def has_audio(self):
        return bool(self.audio_file and self.audio_file.name)

    @property
    def has_transcript(self):
        return self.transcript_segments.exists()

    @property
    def has_summary(self):
        return hasattr(self, 'summary') and self.summary is not None

    @property
    def has_moments(self):
        return self.important_moments.exists()

    @property
    def has_action_items(self):
        return self.action_items.exists()

    @property
    def full_transcript_text(self):
        """
        Combine all segment text into a single continuous transcript.
        """
        return " ".join(seg.text for seg in self.transcript_segments.all())

    @property
    def latest_job(self):
        """Returns the most recent processing job for this video."""
        return self.processing_jobs.first()

    @property
    def status_badge_class(self):
        mapping = {
            self.STATUS_UPLOADED: 'bg-primary-subtle text-primary border border-primary-subtle',
            self.STATUS_QUEUED: 'bg-info-subtle text-info border border-info-subtle',
            self.STATUS_PROCESSING: 'bg-warning-subtle text-warning border border-warning-subtle',
            self.STATUS_COMPLETED: 'bg-success-subtle text-success border border-success-subtle',
            self.STATUS_FAILED: 'bg-danger-subtle text-danger border border-danger-subtle',
        }
        return mapping.get(self.status, 'bg-secondary-subtle text-secondary')


class TranscriptSegment(models.Model):
    """
    Model representing a timestamped segment of the transcribed audio.
    """
    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='transcript_segments',
        help_text="The video this transcript segment belongs to"
    )
    start_time = models.FloatField(help_text="Start time in seconds")
    end_time = models.FloatField(help_text="End time in seconds")
    text = models.TextField(help_text="Transcribed speech text for this timestamp interval")
    segment_order = models.PositiveIntegerField(default=0, help_text="Chronological order index of the segment")

    class Meta:
        ordering = ['segment_order', 'start_time']
        verbose_name = 'Transcript Segment'
        verbose_name_plural = 'Transcript Segments'

    def __str__(self):
        return f"[{self.formatted_start_time} -> {self.formatted_end_time}] {self.text[:50]}"

    @property
    def formatted_start_time(self) -> str:
        """Timestamp formatted as HH:MM:SS"""
        return format_seconds_to_hhmmss(self.start_time)

    @property
    def formatted_end_time(self) -> str:
        """Timestamp formatted as HH:MM:SS"""
        return format_seconds_to_hhmmss(self.end_time)

    @property
    def formatted_time_range(self) -> str:
        """Formatted range e.g. '00:00:02 - 00:00:07'"""
        return f"{self.formatted_start_time} - {self.formatted_end_time}"


class Summary(models.Model):
    """
    Model storing AI-generated structured notes and summaries for a video.
    """
    video = models.OneToOneField(
        Video,
        on_delete=models.CASCADE,
        related_name='summary',
        help_text="The video associated with this summary"
    )
    short_summary = models.TextField(
        help_text="Executive summary / TL;DR (2-3 sentences)"
    )
    detailed_summary = models.TextField(
        help_text="Comprehensive, multi-paragraph detailed summary of the lecture/video content"
    )
    key_points = models.JSONField(
        default=list,
        help_text="List of bulleted key takeaways and learnings"
    )
    main_topics = models.JSONField(
        default=list,
        help_text="List of central topics and themes discussed"
    )
    important_concepts = models.JSONField(
        default=list,
        help_text="List of key concepts, formulas, or definitions explained"
    )
    model_used = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="Name of the LLM model used to generate this summary (e.g. 'gemini-2.5-flash', 'gpt-4o-mini')"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Video Summary'
        verbose_name_plural = 'Video Summaries'

    def __str__(self):
        return f"Summary for {self.video.title} ({self.video.user.username})"


class ImportantMoment(models.Model):
    """
    Model representing an AI-detected key moment or highlighted timestamp in a video.
    """
    IMPORTANCE_CRITICAL = 'critical'
    IMPORTANCE_HIGH = 'high'
    IMPORTANCE_MEDIUM = 'medium'

    IMPORTANCE_CHOICES = [
        (IMPORTANCE_CRITICAL, 'Critical'),
        (IMPORTANCE_HIGH, 'High'),
        (IMPORTANCE_MEDIUM, 'Medium'),
    ]

    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='important_moments',
        help_text="The video this key moment belongs to"
    )
    title = models.CharField(max_length=255, help_text="Short, descriptive title of this moment")
    description = models.TextField(blank=True, help_text="Short explanation of why this moment is significant")
    start_time = models.FloatField(help_text="Start time in seconds")
    end_time = models.FloatField(help_text="End time in seconds")
    importance = models.CharField(
        max_length=20,
        choices=IMPORTANCE_CHOICES,
        default=IMPORTANCE_HIGH,
        help_text="Relevance/Importance priority level"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['start_time', 'created_at']
        verbose_name = 'Important Moment'
        verbose_name_plural = 'Important Moments'

    def __str__(self):
        return f"[{self.formatted_start_time}] {self.title} ({self.get_importance_display()})"

    @property
    def formatted_start_time(self) -> str:
        """Timestamp formatted as HH:MM:SS"""
        return format_seconds_to_hhmmss(self.start_time)

    @property
    def formatted_end_time(self) -> str:
        """Timestamp formatted as HH:MM:SS"""
        return format_seconds_to_hhmmss(self.end_time)

    @property
    def formatted_time_range(self) -> str:
        return f"{self.formatted_start_time} - {self.formatted_end_time}"

    @property
    def importance_badge_class(self) -> str:
        mapping = {
            self.IMPORTANCE_CRITICAL: 'bg-danger-subtle text-danger border border-danger-subtle',
            self.IMPORTANCE_HIGH: 'bg-primary-subtle text-primary border border-primary-subtle',
            self.IMPORTANCE_MEDIUM: 'bg-info-subtle text-info border border-info-subtle',
        }
        return mapping.get(self.importance, 'bg-secondary-subtle text-secondary')


class ActionItem(models.Model):
    """
    Model representing an AI-detected actionable task or assignment extracted from video/lecture/meeting.
    """
    PRIORITY_LOW = 'low'
    PRIORITY_MEDIUM = 'medium'
    PRIORITY_HIGH = 'high'

    PRIORITY_CHOICES = [
        (PRIORITY_LOW, 'Low'),
        (PRIORITY_MEDIUM, 'Medium'),
        (PRIORITY_HIGH, 'High'),
    ]

    STATUS_PENDING = 'pending'
    STATUS_COMPLETED = 'completed'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_COMPLETED, 'Completed'),
    ]

    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='action_items',
        help_text="The video this action item belongs to"
    )
    task = models.CharField(max_length=255, help_text="Short title or description of the actionable task")
    description = models.TextField(blank=True, help_text="Detailed context or notes regarding this action item")
    priority = models.CharField(
        max_length=10,
        choices=PRIORITY_CHOICES,
        default=PRIORITY_MEDIUM,
        help_text="Urgency / Priority level"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
        help_text="Current state of the action item (Pending or Completed)"
    )
    timestamp = models.FloatField(
        null=True,
        blank=True,
        help_text="Time in seconds where the action item or task is discussed in the video"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['status', 'timestamp', '-created_at']
        verbose_name = 'Action Item'
        verbose_name_plural = 'Action Items'

    def __str__(self):
        return f"[{self.get_priority_display()}] {self.task} ({self.get_status_display()})"

    @property
    def formatted_timestamp(self) -> str:
        """Timestamp formatted as HH:MM:SS"""
        if self.timestamp is None:
            return "00:00:00"
        return format_seconds_to_hhmmss(self.timestamp)

    @property
    def is_completed(self) -> bool:
        return self.status == self.STATUS_COMPLETED

    @property
    def priority_badge_class(self) -> str:
        mapping = {
            self.PRIORITY_HIGH: 'bg-danger-subtle text-danger border border-danger-subtle',
            self.PRIORITY_MEDIUM: 'bg-warning-subtle text-warning border border-warning-subtle',
            self.PRIORITY_LOW: 'bg-info-subtle text-info border border-info-subtle',
        }
        return mapping.get(self.priority, 'bg-secondary-subtle text-secondary')

    @property
    def status_badge_class(self) -> str:
        if self.status == self.STATUS_COMPLETED:
            return 'bg-success-subtle text-success border border-success-subtle'
        return 'bg-secondary-subtle text-secondary border border-secondary-subtle'


class ProcessingJob(models.Model):
    """
    Model tracking execution lifecycle, stages, and error logs for the complete video processing pipeline.
    """
    STATUS_QUEUED = 'queued'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_QUEUED, 'Queued'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_COMPLETED, 'Completed'),
        (STATUS_FAILED, 'Failed'),
    ]

    STEP_VALIDATION = 'validation'
    STEP_AUDIO_EXTRACTION = 'audio_extraction'
    STEP_TRANSCRIPTION = 'transcription'
    STEP_SUMMARIZATION = 'summarization'
    STEP_TIMESTAMPS = 'timestamps'
    STEP_ACTION_ITEMS = 'action_items'
    STEP_RAG_INDEXING = 'rag_indexing'
    STEP_COMPLETED = 'completed'

    STEP_CHOICES = [
        (STEP_VALIDATION, 'Validating Video File'),
        (STEP_AUDIO_EXTRACTION, 'Extracting Audio (FFmpeg)'),
        (STEP_TRANSCRIPTION, 'Transcribing Speech (Faster-Whisper)'),
        (STEP_SUMMARIZATION, 'Generating AI Summary & Notes'),
        (STEP_TIMESTAMPS, 'Detecting Important Timestamps'),
        (STEP_ACTION_ITEMS, 'Extracting Action Items & Tasks'),
        (STEP_RAG_INDEXING, 'Vector Indexing (ChromaDB RAG)'),
        (STEP_COMPLETED, 'Pipeline Completed'),
    ]

    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='processing_jobs',
        help_text="The video being processed"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_QUEUED,
        db_index=True
    )
    current_step = models.CharField(
        max_length=30,
        choices=STEP_CHOICES,
        default=STEP_VALIDATION
    )
    progress_percent = models.PositiveSmallIntegerField(
        default=0,
        help_text="Pipeline completion progress (0-100%)"
    )
    error_message = models.TextField(
        blank=True,
        help_text="Error message if pipeline or a step encountered an error"
    )
    error_step = models.CharField(
        max_length=30,
        blank=True,
        help_text="Specific pipeline stage where failure occurred"
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Processing Job'
        verbose_name_plural = 'Processing Jobs'

    def __str__(self):
        return f"Job #{self.pk} for Video '{self.video.title}' [{self.get_status_display()} - {self.get_current_step_display()}]"

    @property
    def is_running(self) -> bool:
        return self.status in (self.STATUS_QUEUED, self.STATUS_PROCESSING)

    @property
    def is_completed(self) -> bool:
        return self.status == self.STATUS_COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == self.STATUS_FAILED

    @property
    def status_badge_class(self) -> str:
        mapping = {
            self.STATUS_QUEUED: 'bg-info-subtle text-info border border-info-subtle',
            self.STATUS_PROCESSING: 'bg-warning-subtle text-warning border border-warning-subtle',
            self.STATUS_COMPLETED: 'bg-success-subtle text-success border border-success-subtle',
            self.STATUS_FAILED: 'bg-danger-subtle text-danger border border-danger-subtle',
        }
        return mapping.get(self.status, 'bg-secondary-subtle text-secondary')


