import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.utils.text import slugify
from .models import Video, TranscriptSegment, Summary, ImportantMoment, ActionItem, ProcessingJob
from .forms import VideoUploadForm
from .services.media import (
    process_video_audio_extraction,
    FFmpegError,
    FFmpegNotFoundError,
    InvalidVideoError,
    AudioExtractionError
)
from .services.transcription import (
    process_video_transcription,
    TranscriptionError,
    AudioFileNotFoundError,
    WhisperModelLoadError
)
from .services.summarization import (
    process_video_summarization,
    SummarizationError,
    EmptyTranscriptError,
    LLMAPIError
)
from .services.timestamps import (
    process_important_moments_detection,
    TimestampDetectionError
)
from .services.action_items import (
    process_action_items_extraction,
    ActionItemExtractionError
)
from .services.pipeline import (
    run_complete_pipeline,
    retry_failed_pipeline,
    PipelineError
)


@login_required
def dashboard_view(request):
    """
    User dashboard displaying only the authenticated user's uploaded videos.
    """
    user_videos = Video.objects.filter(user=request.user)
    
    total_count = user_videos.count()
    completed_count = user_videos.filter(status=Video.STATUS_COMPLETED).count()
    processing_count = user_videos.filter(status__in=[Video.STATUS_PROCESSING, Video.STATUS_QUEUED]).count()
    uploaded_count = user_videos.filter(status=Video.STATUS_UPLOADED).count()

    context = {
        'videos': user_videos,
        'stats': {
            'total': total_count,
            'completed': completed_count,
            'processing': processing_count,
            'uploaded': uploaded_count,
        }
    }
    return render(request, 'videos/dashboard.html', context)


@login_required
def upload_view(request):
    """
    Handle video file upload and automatically execute the complete processing pipeline:
    1. Validation -> 2. Audio Extraction -> 3. Whisper STT -> 4. AI Summary -> 5. Timestamps -> 6. Action Items -> 7. ChromaDB RAG Indexing
    """
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            video.save()

            try:
                # Run the complete unified pipeline
                run_complete_pipeline(video, async_mode=False)
                moments_count = video.important_moments.count()
                action_items_count = video.action_items.count()
                messages.success(
                    request,
                    f"Video '{video.title}' successfully processed: speech transcribed, smart notes generated, {moments_count} moments indexed, {action_items_count} action items extracted, and RAG vector search activated!"
                )
            except Exception as e:
                messages.warning(request, f"Video uploaded, but processing encountered an issue: {str(e)}")

            return redirect('videos:detail', pk=video.pk)
        else:
            messages.error(request, "Error uploading video. Please check the errors below.")
    else:
        form = VideoUploadForm()

    return render(request, 'videos/upload.html', {'form': form})


@login_required
def detail_view(request, pk):
    """
    Display video details, media player, extracted audio, transcript segments, summary overview,
    interactive key moments, action items preview, and processing pipeline lifecycle.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    transcript_segments = video.transcript_segments.all()
    summary = getattr(video, 'summary', None)
    important_moments = video.important_moments.all()
    action_items = video.action_items.all()
    latest_job = video.latest_job

    context = {
        'video': video,
        'summary': summary,
        'transcript_segments': transcript_segments,
        'important_moments': important_moments,
        'action_items': action_items,
        'latest_job': latest_job,
        'has_transcript': transcript_segments.exists(),
        'has_summary': summary is not None,
        'has_moments': important_moments.exists(),
        'has_action_items': action_items.exists(),
        'segment_count': transcript_segments.count(),
        'moments_count': important_moments.count(),
        'action_items_count': action_items.count(),
        'pending_action_items_count': action_items.filter(status=ActionItem.STATUS_PENDING).count(),
    }
    return render(request, 'videos/detail.html', context)


@login_required
def process_pipeline_view(request, pk):
    """
    Manually trigger or retry the complete end-to-end processing pipeline for a video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    if request.method == 'POST':
        try:
            run_complete_pipeline(video, async_mode=False, force_reprocess=True)
            messages.success(request, f"Complete processing pipeline re-executed successfully for '{video.title}'!")
        except Exception as e:
            messages.error(request, f"Pipeline execution failed: {str(e)}")

        return redirect('videos:detail', pk=video.pk)
    
    return redirect('videos:detail', pk=video.pk)


@login_required
def processing_status_api_view(request, pk):
    """
    API endpoint returning JSON status of video processing pipeline and latest job.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    job = video.latest_job
    data = {
        'video_id': video.pk,
        'title': video.title,
        'status': video.status,
        'status_display': video.get_status_display(),
        'error_message': video.error_message,
        'has_audio': video.has_audio,
        'has_transcript': video.has_transcript,
        'has_summary': video.has_summary,
        'has_moments': video.has_moments,
        'has_action_items': video.has_action_items,
        'job': {
            'id': job.pk if job else None,
            'status': job.status if job else video.status,
            'current_step': job.current_step if job else '',
            'current_step_display': job.get_current_step_display() if job else '',
            'progress_percent': job.progress_percent if job else 0,
            'error_message': job.error_message if job else '',
            'error_step': job.error_step if job else '',
        } if job else None
    }
    return JsonResponse(data)


@login_required
def action_items_view(request, pk):
    """
    Dedicated Action Items page for a specific video.
    Allows filtering by status and priority, marking items complete, editing priority, and creating tasks.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    
    # Auto-extract if empty and transcript exists
    if not video.action_items.exists() and video.has_transcript:
        try:
            process_action_items_extraction(video)
        except Exception as e:
            messages.warning(request, f"Could not auto-extract action items: {str(e)}")

    items_qs = video.action_items.all()

    # Filters
    status_filter = request.GET.get('status', 'all').lower()
    priority_filter = request.GET.get('priority', 'all').lower()

    if status_filter in ('pending', 'completed'):
        items_qs = items_qs.filter(status=status_filter)

    if priority_filter in ('low', 'medium', 'high'):
        items_qs = items_qs.filter(priority=priority_filter)

    all_items = video.action_items.all()
    stats = {
        'total': all_items.count(),
        'completed': all_items.filter(status=ActionItem.STATUS_COMPLETED).count(),
        'pending': all_items.filter(status=ActionItem.STATUS_PENDING).count(),
        'high': all_items.filter(priority=ActionItem.PRIORITY_HIGH).count(),
    }

    context = {
        'video': video,
        'action_items': items_qs,
        'stats': stats,
        'current_status': status_filter,
        'current_priority': priority_filter,
        'has_action_items': all_items.exists(),
    }
    return render(request, 'videos/action_items.html', context)


@login_required
def extract_action_items_action_view(request, pk):
    """
    Manually trigger or re-extract actionable tasks from video transcript.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        try:
            items = process_action_items_extraction(video)
            messages.success(request, f"Extracted {len(items)} actionable tasks and follow-ups!")
        except Exception as e:
            messages.error(request, f"Action item extraction failed: {str(e)}")

    return redirect('videos:action_items', pk=video.pk)


@login_required
def toggle_action_item_status_view(request, pk, item_id):
    """
    Toggle an action item between Pending and Completed status. Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    item = get_object_or_404(ActionItem, pk=item_id, video=video)

    if request.method == 'POST':
        if item.status == ActionItem.STATUS_COMPLETED:
            item.status = ActionItem.STATUS_PENDING
            msg = f"Task '{item.task}' marked as Pending."
        else:
            item.status = ActionItem.STATUS_COMPLETED
            msg = f"Task '{item.task}' marked as Completed!"
        item.save()
        messages.success(request, msg)

    next_url = request.POST.get('next') or request.GET.get('next')
    if next_url:
        return redirect(next_url)
    return redirect('videos:action_items', pk=video.pk)


@login_required
def update_action_item_priority_view(request, pk, item_id):
    """
    Update priority of an action item (low, medium, high). Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    item = get_object_or_404(ActionItem, pk=item_id, video=video)

    if request.method == 'POST':
        new_priority = request.POST.get('priority', '').strip().lower()
        if new_priority in (ActionItem.PRIORITY_LOW, ActionItem.PRIORITY_MEDIUM, ActionItem.PRIORITY_HIGH):
            item.priority = new_priority
            item.save()
            messages.success(request, f"Priority for '{item.task}' updated to {item.get_priority_display()}.")
        else:
            messages.error(request, "Invalid priority level specified.")

    next_url = request.POST.get('next') or request.GET.get('next')
    if next_url:
        return redirect(next_url)
    return redirect('videos:action_items', pk=video.pk)


@login_required
def delete_action_item_view(request, pk, item_id):
    """
    Delete an action item. Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    item = get_object_or_404(ActionItem, pk=item_id, video=video)

    if request.method == 'POST':
        task_name = item.task
        item.delete()
        messages.success(request, f"Task '{task_name}' removed successfully.")

    next_url = request.POST.get('next') or request.GET.get('next')
    if next_url:
        return redirect(next_url)
    return redirect('videos:action_items', pk=video.pk)


@login_required
def create_action_item_view(request, pk):
    """
    Manually add a new action item to a video. Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        task = request.POST.get('task', '').strip()
        description = request.POST.get('description', '').strip()
        priority = request.POST.get('priority', ActionItem.PRIORITY_MEDIUM).lower()
        raw_timestamp = request.POST.get('timestamp', '').strip()

        if not task:
            messages.error(request, "Task title is required.")
            return redirect('videos:action_items', pk=video.pk)

        if priority not in (ActionItem.PRIORITY_LOW, ActionItem.PRIORITY_MEDIUM, ActionItem.PRIORITY_HIGH):
            priority = ActionItem.PRIORITY_MEDIUM

        timestamp = None
        if raw_timestamp:
            try:
                timestamp = float(raw_timestamp)
            except ValueError:
                pass

        ActionItem.objects.create(
            video=video,
            task=task,
            description=description,
            priority=priority,
            status=ActionItem.STATUS_PENDING,
            timestamp=timestamp
        )
        messages.success(request, f"New action item '{task}' added successfully.")

    return redirect('videos:action_items', pk=video.pk)



@login_required
def detect_moments_action_view(request, pk):
    """
    Manually trigger or re-detect important timestamps and key moments for a video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        try:
            moments = process_important_moments_detection(video)
            messages.success(request, f"Detected {len(moments)} key moments and chapter highlights!")
        except Exception as e:
            messages.error(request, f"Key moment detection failed: {str(e)}")

    return redirect('videos:detail', pk=video.pk)


@login_required
def summary_view(request, pk):
    """
    Dedicated study notes and AI Summary page for a specific video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    
    summary = getattr(video, 'summary', None)
    if not summary and video.has_transcript:
        try:
            summary = process_video_summarization(video)
            messages.success(request, "AI Summary and study notes generated successfully!")
        except Exception as e:
            messages.warning(request, f"Could not generate summary: {str(e)}")

    context = {
        'video': video,
        'summary': summary,
        'has_summary': summary is not None,
    }
    return render(request, 'videos/summary.html', context)


@login_required
def generate_summary_action_view(request, pk):
    """
    Manually trigger or re-generate AI summary notes for a video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        try:
            process_video_summarization(video)
            messages.success(request, "AI Summary notes regenerated successfully!")
        except Exception as e:
            messages.error(request, f"Summarization failed: {str(e)}")

    return redirect('videos:summary', pk=video.pk)


@login_required
def extract_audio_action_view(request, pk):
    """
    Manually trigger or retry audio extraction for an existing video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        try:
            process_video_audio_extraction(video)
            messages.success(
                request,
                f"Audio extracted successfully! Duration: {video.formatted_duration}."
            )
        except Exception as e:
            messages.error(request, f"Audio extraction failed: {str(e)}")

    return redirect('videos:detail', pk=video.pk)


@login_required
def transcribe_action_view(request, pk):
    """
    Manually trigger or re-run speech-to-text transcription for an existing video.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        try:
            process_video_transcription(video)
            segment_count = video.transcript_segments.count()
            messages.success(
                request,
                f"Transcription complete! Generated {segment_count} timestamped segments (Language: {video.language.upper()})."
            )
        except Exception as e:
            messages.error(request, f"Transcription failed: {str(e)}")

    return redirect('videos:detail', pk=video.pk)


@login_required
def delete_view(request, pk):
    """
    Delete a video, associated audio file, transcripts, moments, and summaries owned by the user.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        title = video.title
        # Delete the physical video file if it exists
        if video.video_file and os.path.isfile(video.video_file.path):
            try:
                os.remove(video.video_file.path)
            except OSError:
                pass

        # Delete the physical audio file if it exists
        if video.audio_file and os.path.isfile(video.audio_file.path):
            try:
                os.remove(video.audio_file.path)
            except OSError:
                pass
        
        video.delete()
        messages.success(request, f"Video '{title}' and all its data were deleted successfully.")
        return redirect('videos:dashboard')

    return render(request, 'videos/delete_confirm.html', {'video': video})


@login_required
def download_pdf_report_view(request, pk):
    """
    Dynamically generate and download a publication-quality PDF report for a video.
    Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)
    include_transcript = request.GET.get('transcript', '1') in ('1', 'true', 'yes')

    try:
        from reports.pdf import generate_video_pdf_report, PDFGenerationError
        pdf_buffer = generate_video_pdf_report(video, include_transcript=include_transcript)
        safe_filename = slugify(video.title or f"video_{video.pk}")[:50] or f"video_{video.pk}"
        response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="VidNote_Report_{safe_filename}.pdf"'
        return response
    except Exception as e:
        messages.error(request, f"Could not generate PDF report: {str(e)}")
        return redirect('videos:detail', pk=video.pk)

