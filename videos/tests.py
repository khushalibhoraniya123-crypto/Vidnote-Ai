import os
import tempfile
import shutil
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from .models import (
    Video,
    TranscriptSegment,
    Summary,
    ImportantMoment,
    ActionItem,
    ProcessingJob,
    format_seconds_to_hhmmss
)
from .services.media import (
    extract_audio_from_video,
    process_video_audio_extraction,
    get_ffmpeg_path,
    InvalidVideoError
)
from .services.transcription import (
    transcribe_audio_file,
    process_video_transcription,
    AudioFileNotFoundError
)
from .services.summarization import (
    generate_summary_from_transcript,
    process_video_summarization,
    chunk_transcript_if_needed,
    extract_json_from_response
)
from .services.timestamps import (
    deduplicate_and_validate_moments,
    detect_moments_heuristic_fallback,
    process_important_moments_detection
)
from .services.action_items import (
    deduplicate_and_validate_action_items,
    extract_action_items_heuristic_fallback,
    process_action_items_extraction
)

class VideoAppAndTimestampsTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_media_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_media_dir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        self.user_a = User.objects.create_user(username='usera', password='Password123!')
        self.user_b = User.objects.create_user(username='userb', password='Password123!')
        
        self.client_a = Client()
        self.client_a.login(username='usera', password='Password123!')

        self.client_b = Client()
        self.client_b.login(username='userb', password='Password123!')

        self.unauth_client = Client()

    def _get_sample_video_with_audio_file(self):
        sample_path = os.path.join(settings.BASE_DIR, 'test_sample_with_audio.mp4')
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                return SimpleUploadedFile(
                    name="sample_lecture_with_audio.mp4",
                    content=f.read(),
                    content_type="video/mp4"
                )
        return SimpleUploadedFile(
            name="fallback_lecture.mp4",
            content=b"\x00\x00\x00 ftypisom",
            content_type="video/mp4"
        )

    def test_important_moment_model_properties(self):
        """Test ImportantMoment time formatting and properties"""
        video = Video.objects.create(
            user=self.user_a,
            title="Database Normalization Lecture"
        )
        moment = ImportantMoment.objects.create(
            video=video,
            title="Third Normal Form (3NF)",
            description="Detailed breakdown of transitive dependencies.",
            start_time=75.5,
            end_time=130.0,
            importance=ImportantMoment.IMPORTANCE_CRITICAL
        )
        self.assertEqual(moment.formatted_start_time, "00:01:15")
        self.assertEqual(moment.formatted_end_time, "00:02:10")
        self.assertEqual(moment.formatted_time_range, "00:01:15 - 00:02:10")
        self.assertIn("danger", moment.importance_badge_class)
        self.assertEqual(video.important_moments.count(), 1)
        self.assertTrue(video.has_moments)

    def test_deduplicate_and_validate_moments(self):
        """Test timestamp deduplication and validation"""
        raw_moments = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 10.0, 'importance': 'high'},
            {'title': 'Duplicate Intro', 'start_time': 2.0, 'end_time': 12.0, 'importance': 'high'},  # within 4s threshold -> duplicate
            {'title': 'Deep Dive', 'start_time': 45.0, 'end_time': 60.0, 'importance': 'critical'},
        ]
        deduped = deduplicate_and_validate_moments(raw_moments)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]['title'], 'Intro')
        self.assertEqual(deduped[1]['title'], 'Deep Dive')

    def test_process_important_moments_detection(self):
        """Test process_important_moments_detection service generates and stores moments"""
        video = Video.objects.create(
            user=self.user_a,
            title="Machine Learning 101"
        )
        TranscriptSegment.objects.create(
            video=video,
            start_time=0.0,
            end_time=10.0,
            text="Welcome to machine learning. Today we study gradient descent algorithms.",
            segment_order=0
        )
        TranscriptSegment.objects.create(
            video=video,
            start_time=10.0,
            end_time=25.0,
            text="Let us define the loss function formula and learning rate parameter.",
            segment_order=1
        )

        moments = process_important_moments_detection(video)
        self.assertGreater(len(moments), 0)
        self.assertTrue(video.has_moments)
        self.assertEqual(video.important_moments.first().video, video)

    def test_manual_detect_moments_action_view(self):
        """Test manual detect moments POST action"""
        video = Video.objects.create(
            user=self.user_a,
            title="Algorithm Analysis"
        )
        TranscriptSegment.objects.create(
            video=video,
            start_time=0.0,
            end_time=5.0,
            text="Time complexity and Big O notation discussion.",
            segment_order=0
        )

        response = self.client_a.post(
            reverse('videos:detect_moments', kwargs={'pk': video.pk}),
            follow=True
        )
        self.assertEqual(response.status_code, 200)
        video.refresh_from_db()
        self.assertTrue(video.has_moments)

    def test_cascade_deletion_deletes_important_moments(self):
        """Deleting a video cascades and removes all its important moments"""
        video = Video.objects.create(
            user=self.user_a,
            title="Temporary Video"
        )
        ImportantMoment.objects.create(
            video=video,
            title="Key Timestamp",
            start_time=0.0,
            end_time=5.0
        )
        self.assertEqual(ImportantMoment.objects.count(), 1)

        video.delete()
        self.assertEqual(ImportantMoment.objects.count(), 0)

    def test_user_isolation_video_detail_and_moments(self):
        """User B cannot see User A's video or key moments"""
        video_a = Video.objects.create(
            user=self.user_a,
            title="User A Private Timestamp Video"
        )
        ImportantMoment.objects.create(
            video=video_a,
            title="Secret Moment",
            description="Classified lecture topic",
            start_time=5.0,
            end_time=15.0
        )

        # User B cannot access User A's detail page (404)
        response_b = self.client_b.get(reverse('videos:detail', kwargs={'pk': video_a.pk}))
        self.assertEqual(response_b.status_code, 404)

        # User A can view their moment and jump timestamp
        response_a = self.client_a.get(reverse('videos:detail', kwargs={'pk': video_a.pk}))
        self.assertEqual(response_a.status_code, 200)
        self.assertContains(response_a, "Secret Moment")
        self.assertContains(response_a, "00:00:05")

    # =========================================================================
    # STEP 8: ACTION ITEM EXTRACTION & MANAGEMENT TESTS
    # =========================================================================

    def test_action_item_model_properties_and_defaults(self):
        """Test ActionItem model attributes, formatting, and status helpers"""
        video = Video.objects.create(
            user=self.user_a,
            title="Sprint Planning Meeting"
        )
        item = ActionItem.objects.create(
            video=video,
            task="Submit API documentation for sprint review",
            description="Make sure swagger annotations are complete by Friday.",
            priority=ActionItem.PRIORITY_HIGH,
            status=ActionItem.STATUS_PENDING,
            timestamp=92.5
        )
        self.assertEqual(item.formatted_timestamp, "00:01:32")
        self.assertFalse(item.is_completed)
        self.assertIn("danger", item.priority_badge_class)
        self.assertIn("secondary", item.status_badge_class)

        # Mark completed
        item.status = ActionItem.STATUS_COMPLETED
        item.save()
        self.assertTrue(item.is_completed)
        self.assertIn("success", item.status_badge_class)
        self.assertTrue(video.has_action_items)

    def test_deduplicate_and_validate_action_items(self):
        """Test action item validation, deduplication, and priority fallback"""
        raw_items = [
            {'task': 'Prepare slides', 'priority': 'high', 'status': 'pending', 'timestamp': 10.0},
            {'task': 'Prepare slides', 'priority': 'medium', 'status': 'pending', 'timestamp': 12.0},  # duplicate
            {'task': 'Review code PR', 'priority': 'invalid_prio', 'status': 'pending', 'timestamp': 45.0},
            {'task': '', 'priority': 'low'},  # empty task -> skip
        ]
        deduped = deduplicate_and_validate_action_items(raw_items, max_duration=60.0)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]['task'], 'Prepare slides')
        self.assertEqual(deduped[0]['priority'], 'high')
        self.assertEqual(deduped[1]['task'], 'Review code PR')
        self.assertEqual(deduped[1]['priority'], 'medium')  # invalid normalized to medium

    def test_extract_action_items_heuristic_meeting_transcript(self):
        """Test intelligent heuristic task extraction with meeting-style dialogue"""
        video = Video.objects.create(
            user=self.user_a,
            title="Q3 Engineering Sync"
        )
        seg1 = TranscriptSegment.objects.create(
            video=video,
            start_time=0.0,
            end_time=15.0,
            text="Good morning team. We must immediately fix the urgent blocker on the authentication service.",
            segment_order=0
        )
        seg2 = TranscriptSegment.objects.create(
            video=video,
            start_time=16.0,
            end_time=35.0,
            text="Also, please make sure to review the database migration PR and test the deployment script.",
            segment_order=1
        )
        seg3 = TranscriptSegment.objects.create(
            video=video,
            start_time=36.0,
            end_time=50.0,
            text="If time permits, we can consider looking into optional UI themes for dark mode.",
            segment_order=2
        )

        segments = [seg1, seg2, seg3]
        items = extract_action_items_heuristic_fallback(segments, video.title, video_duration=60.0)
        
        self.assertGreaterEqual(len(items), 2)
        # Verify priority assignment logic
        priorities = [it['priority'] for it in items]
        self.assertIn('high', priorities)

    def test_process_action_items_extraction_service(self):
        """Test full process_action_items_extraction pipeline stores tasks and preserves completed items"""
        video = Video.objects.create(
            user=self.user_a,
            title="Team All Hands"
        )
        TranscriptSegment.objects.create(
            video=video,
            start_time=5.0,
            end_time=20.0,
            text="Action item: Alice needs to prepare and submit the quarterly performance report.",
            segment_order=0
        )

        # 1. First extraction
        items = process_action_items_extraction(video)
        self.assertGreater(len(items), 0)
        self.assertEqual(video.action_items.count(), len(items))

        # Mark first item as completed
        first_item = video.action_items.first()
        first_item.status = ActionItem.STATUS_COMPLETED
        first_item.save()

        # 2. Re-extraction should preserve completed status for this task
        process_action_items_extraction(video)
        re_extracted_first = video.action_items.filter(task=first_item.task).first()
        self.assertIsNotNone(re_extracted_first)
        self.assertEqual(re_extracted_first.status, ActionItem.STATUS_COMPLETED)

    def test_toggle_action_item_status_view(self):
        """Test POST toggle status between Pending and Completed"""
        video = Video.objects.create(
            user=self.user_a,
            title="Product Backlog Refinement"
        )
        item = ActionItem.objects.create(
            video=video,
            task="Refactor checkout API",
            status=ActionItem.STATUS_PENDING
        )

        url = reverse('videos:toggle_action_item_status', kwargs={'pk': video.pk, 'item_id': item.pk})
        
        # Toggle to Completed
        res = self.client_a.post(url, follow=True)
        self.assertEqual(res.status_code, 200)
        item.refresh_from_db()
        self.assertEqual(item.status, ActionItem.STATUS_COMPLETED)

        # Toggle back to Pending
        res2 = self.client_a.post(url, follow=True)
        self.assertEqual(res2.status_code, 200)
        item.refresh_from_db()
        self.assertEqual(item.status, ActionItem.STATUS_PENDING)

    def test_update_action_item_priority_view(self):
        """Test POST updating priority of an action item"""
        video = Video.objects.create(
            user=self.user_a,
            title="Security Audit"
        )
        item = ActionItem.objects.create(
            video=video,
            task="Patch dependencies",
            priority=ActionItem.PRIORITY_LOW
        )

        url = reverse('videos:update_action_item_priority', kwargs={'pk': video.pk, 'item_id': item.pk})
        res = self.client_a.post(url, {'priority': 'high'}, follow=True)
        self.assertEqual(res.status_code, 200)
        item.refresh_from_db()
        self.assertEqual(item.priority, ActionItem.PRIORITY_HIGH)

    def test_create_and_delete_custom_action_item_views(self):
        """Test manual creation and deletion of action items"""
        video = Video.objects.create(
            user=self.user_a,
            title="Client Kickoff"
        )
        create_url = reverse('videos:create_action_item', kwargs={'pk': video.pk})
        res = self.client_a.post(create_url, {
            'task': 'Send follow-up email to client',
            'description': 'Include contract PDF and project timeline.',
            'priority': 'high',
            'timestamp': '15.5'
        }, follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(video.action_items.count(), 1)
        created_item = video.action_items.first()
        self.assertEqual(created_item.task, 'Send follow-up email to client')
        self.assertEqual(created_item.timestamp, 15.5)

        # Delete the action item
        delete_url = reverse('videos:delete_action_item', kwargs={'pk': video.pk, 'item_id': created_item.pk})
        del_res = self.client_a.post(delete_url, follow=True)
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(video.action_items.count(), 0)

    def test_action_items_user_isolation_permissions(self):
        """User B cannot access or modify User A's action items"""
        video_a = Video.objects.create(
            user=self.user_a,
            title="Confidential Executive Briefing"
        )
        item_a = ActionItem.objects.create(
            video=video_a,
            task="Restructure department budgets",
            priority=ActionItem.PRIORITY_HIGH
        )

        # User B cannot view User A's action items page
        b_view_res = self.client_b.get(reverse('videos:action_items', kwargs={'pk': video_a.pk}))
        self.assertEqual(b_view_res.status_code, 404)

        # User B cannot toggle User A's action item
        b_toggle_res = self.client_b.post(
            reverse('videos:toggle_action_item_status', kwargs={'pk': video_a.pk, 'item_id': item_a.pk})
        )
        self.assertEqual(b_toggle_res.status_code, 404)

        # User B cannot change priority
        b_prio_res = self.client_b.post(
            reverse('videos:update_action_item_priority', kwargs={'pk': video_a.pk, 'item_id': item_a.pk}),
            {'priority': 'low'}
        )
        self.assertEqual(b_prio_res.status_code, 404)

        # User B cannot delete User A's action item
        b_del_res = self.client_b.post(
            reverse('videos:delete_action_item', kwargs={'pk': video_a.pk, 'item_id': item_a.pk})
        )
        self.assertEqual(b_del_res.status_code, 404)

        # Confirm item still exists untouched
        self.assertEqual(ActionItem.objects.count(), 1)

    def test_processing_job_model_and_properties(self):
        """Test ProcessingJob model creation, choices, and state helper properties."""
        video = Video.objects.create(
            user=self.user_a,
            title="Operating Systems: Processes & Threads"
        )
        job = ProcessingJob.objects.create(
            video=video,
            status=ProcessingJob.STATUS_PROCESSING,
            current_step=ProcessingJob.STEP_TRANSCRIPTION,
            progress_percent=45
        )
        self.assertTrue(job.is_running)
        self.assertFalse(job.is_completed)
        self.assertFalse(job.is_failed)
        self.assertIn("warning", job.status_badge_class)
        self.assertEqual(video.latest_job, job)

        # Mark completed
        job.status = ProcessingJob.STATUS_COMPLETED
        job.progress_percent = 100
        job.save()
        self.assertTrue(job.is_completed)
        self.assertFalse(job.is_running)
        self.assertIn("success", job.status_badge_class)

    def test_complete_pipeline_execution_with_sample_video(self):
        """
        Test the complete end-to-end video pipeline workflow:
        Validate -> Extract Audio -> Whisper Transcription -> AI Summary -> Important Timestamps -> Action Items -> RAG Indexing -> Complete.
        """
        uploaded_file = self._get_sample_video_with_audio_file()
        video = Video.objects.create(
            user=self.user_a,
            title="Introduction to Machine Learning",
            video_file=uploaded_file
        )

        from .services.pipeline import run_complete_pipeline
        job = run_complete_pipeline(video, async_mode=False)

        video.refresh_from_db()
        job.refresh_from_db()

        # Check job and video state
        self.assertEqual(job.status, ProcessingJob.STATUS_COMPLETED)
        self.assertEqual(job.progress_percent, 100)
        self.assertEqual(video.status, Video.STATUS_COMPLETED)

        # Check that all artifacts were created
        self.assertTrue(video.has_audio)
        self.assertTrue(video.has_transcript)
        self.assertTrue(video.has_summary)
        self.assertTrue(video.has_moments)
        self.assertTrue(video.has_action_items)

        # Check RAG vector store search
        from rag.services import semantic_search_video
        results = semantic_search_video(video, query="machine learning", top_k=3)
        self.assertIsInstance(results, list)

    def test_pipeline_duplicate_prevention(self):
        """Test that calling run_complete_pipeline on an active job avoids duplicate execution."""
        video = Video.objects.create(
            user=self.user_a,
            title="Database Concurrency"
        )
        from django.utils import timezone
        active_job = ProcessingJob.objects.create(
            video=video,
            status=ProcessingJob.STATUS_PROCESSING,
            current_step=ProcessingJob.STEP_AUDIO_EXTRACTION,
            started_at=timezone.now(),
            progress_percent=25
        )

        from .services.pipeline import run_complete_pipeline
        second_job = run_complete_pipeline(video, async_mode=False, force_reprocess=False)
        self.assertEqual(active_job.pk, second_job.pk)
        self.assertEqual(video.processing_jobs.count(), 1)

    def test_pipeline_error_handling_on_invalid_file(self):
        """Test pipeline graceful failure and error tracking on corrupt or empty video."""
        empty_file = SimpleUploadedFile("empty.mp4", b"", content_type="video/mp4")
        video = Video.objects.create(
            user=self.user_a,
            title="Corrupt Video Test",
            video_file=empty_file
        )

        from .services.pipeline import run_complete_pipeline
        with self.assertRaises(Exception):
            run_complete_pipeline(video, async_mode=False)

        video.refresh_from_db()
        job = video.latest_job
        self.assertIsNotNone(job)
        self.assertEqual(job.status, ProcessingJob.STATUS_FAILED)
        self.assertEqual(video.status, Video.STATUS_FAILED)
        self.assertTrue(len(video.error_message) > 0)

    def test_pipeline_retry_and_api_endpoints(self):
        """Test pipeline re-run view, status API, and user authorization isolation."""
        uploaded_file = self._get_sample_video_with_audio_file()
        video_a = Video.objects.create(
            user=self.user_a,
            title="Calculus Optimization",
            video_file=uploaded_file
        )

        # Trigger pipeline via process view
        process_url = reverse('videos:process_pipeline', kwargs={'pk': video_a.pk})
        res = self.client_a.post(process_url, follow=True)
        self.assertEqual(res.status_code, 200)

        video_a.refresh_from_db()
        self.assertEqual(video_a.status, Video.STATUS_COMPLETED)

        # Check status JSON API
        status_url = reverse('videos:processing_status', kwargs={'pk': video_a.pk})
        api_res = self.client_a.get(status_url)
        self.assertEqual(api_res.status_code, 200)
        data = api_res.json()
        self.assertEqual(data['status'], 'completed')
        self.assertTrue(data['has_transcript'])
        self.assertTrue(data['has_summary'])
        self.assertEqual(data['job']['progress_percent'], 100)

        # User B permission isolation: 404
        b_api_res = self.client_b.get(status_url)
        self.assertEqual(b_api_res.status_code, 404)
        b_proc_res = self.client_b.post(process_url)
        self.assertEqual(b_proc_res.status_code, 404)

    def test_upload_form_security_file_validation(self):
        """Test upload validation rejects invalid file extensions, empty files, and excessive sizes."""
        upload_url = reverse('videos:upload')

        # 1. Reject invalid file extension (.exe)
        invalid_ext_file = SimpleUploadedFile("malicious.exe", b"binary content", content_type="application/octet-stream")
        res_ext = self.client_a.post(upload_url, {
            'title': 'Bad File Test',
            'video_file': invalid_ext_file
        })
        self.assertEqual(res_ext.status_code, 200)
        self.assertContains(res_ext, "Unsupported file format")

        # 2. Reject empty file (0 bytes)
        empty_file = SimpleUploadedFile("empty.mp4", b"", content_type="video/mp4")
        res_empty = self.client_a.post(upload_url, {
            'title': 'Empty File Test',
            'video_file': empty_file
        })
        self.assertEqual(res_empty.status_code, 200)
        self.assertContains(res_empty, "The submitted file is empty")

    def test_cross_user_security_and_unauthorized_isolation(self):
        """
        Verify strict authorization: User B receives 404 when attempting to view, delete,
        download PDF, view summary, detect moments, or trigger actions on User A's videos.
        """
        uploaded_file = self._get_sample_video_with_audio_file()
        video_a = Video.objects.create(
            user=self.user_a,
            title="Confidential Research Lecture",
            video_file=uploaded_file
        )

        # 1. Summary page isolation
        res_sum = self.client_b.get(reverse('videos:summary', kwargs={'pk': video_a.pk}))
        self.assertEqual(res_sum.status_code, 404)

        # 2. Generate summary action isolation
        res_gen_sum = self.client_b.post(reverse('videos:generate_summary', kwargs={'pk': video_a.pk}))
        self.assertEqual(res_gen_sum.status_code, 404)

        # 3. PDF download isolation
        res_pdf = self.client_b.get(reverse('videos:download_pdf', kwargs={'pk': video_a.pk}))
        self.assertEqual(res_pdf.status_code, 404)

        # 4. Detect moments isolation
        res_moments = self.client_b.post(reverse('videos:detect_moments', kwargs={'pk': video_a.pk}))
        self.assertEqual(res_moments.status_code, 404)

        # 5. Video delete isolation
        res_del = self.client_b.post(reverse('videos:delete', kwargs={'pk': video_a.pk}))
        self.assertEqual(res_del.status_code, 404)
        self.assertEqual(Video.objects.filter(pk=video_a.pk).count(), 1)

        # 6. User A can successfully delete own video
        res_a_del = self.client_a.post(reverse('videos:delete', kwargs={'pk': video_a.pk}), follow=True)
        self.assertEqual(res_a_del.status_code, 200)
        self.assertEqual(Video.objects.filter(pk=video_a.pk).count(), 0)



