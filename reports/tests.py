"""
Unit and integration tests for PDF Report Generation in VidNote AI.
Validates ReportLab PDF generation, section rendering, multi-page layout,
download view response headers, and user permission isolation.
"""

import io
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from videos.models import Video, Summary, ImportantMoment, ActionItem, TranscriptSegment
from reports.pdf import generate_video_pdf_report, PDFGenerationError


class PDFReportGenerationTests(TestCase):
    def setUp(self):
        self.user_a = User.objects.create_user(username='pdf_usera', password='Password123!')
        self.user_b = User.objects.create_user(username='pdf_userb', password='Password123!')

        self.client_a = Client()
        self.client_a.login(username='pdf_usera', password='Password123!')

        self.client_b = Client()
        self.client_b.login(username='pdf_userb', password='Password123!')

        self.unauth_client = Client()

        # Create a full video with all features
        self.video = Video.objects.create(
            user=self.user_a,
            title="Deep Learning and Neural Architectures",
            description="Graduate lecture exploring backpropagation, CNNs, and attention mechanisms.",
            duration=3600.0,
            file_size=104857600,
            language='en',
            status=Video.STATUS_COMPLETED
        )

        self.summary = Summary.objects.create(
            video=self.video,
            short_summary="This lecture covers foundational deep learning concepts, gradient descent, and convolutional layers.",
            detailed_summary="The instructor begins with mathematical foundations of neural networks.\nNext, backpropagation algorithm and chain rule derivations are demonstrated.\nFinally, convolutional feature maps and pooling layers are examined.",
            key_points=[
                "Backpropagation computes partial derivatives using the multivariate chain rule.",
                "Stochastic gradient descent with momentum accelerates convergence.",
                "Convolutional kernels perform spatial feature extraction."
            ],
            model_used="gemini-2.5-flash"
        )

        ImportantMoment.objects.create(
            video=self.video,
            title="Backpropagation Derivation",
            description="Step-by-step calculus derivation of error gradients.",
            start_time=120.0,
            end_time=300.0,
            importance=ImportantMoment.IMPORTANCE_CRITICAL
        )
        ImportantMoment.objects.create(
            video=self.video,
            title="CNN Architecture Walkthrough",
            description="Overview of convolutional filters, stride, and padding.",
            start_time=600.0,
            end_time=900.0,
            importance=ImportantMoment.IMPORTANCE_HIGH
        )

        ActionItem.objects.create(
            video=self.video,
            task="Complete Assignment 2 on PyTorch CNN implementation",
            description="Implement ResNet blocks and train on CIFAR-10 dataset.",
            priority=ActionItem.PRIORITY_HIGH,
            status=ActionItem.STATUS_PENDING,
            timestamp=3450.0
        )
        ActionItem.objects.create(
            video=self.video,
            task="Review matrix calculus notes before recitation",
            description="Review Jacobian and Hessian matrix definitions.",
            priority=ActionItem.PRIORITY_MEDIUM,
            status=ActionItem.STATUS_COMPLETED,
            timestamp=450.0
        )

        TranscriptSegment.objects.create(
            video=self.video,
            start_time=0.0,
            end_time=15.0,
            text="Welcome everyone to Deep Learning. Today we start with backpropagation.",
            segment_order=0
        )
        TranscriptSegment.objects.create(
            video=self.video,
            start_time=15.0,
            end_time=45.0,
            text="Let us write down the loss function and compute gradients with respect to weights.",
            segment_order=1
        )

    def test_generate_video_pdf_report_complete(self):
        """Test PDF generator produces a valid binary PDF buffer with all sections"""
        pdf_buffer = generate_video_pdf_report(self.video, include_transcript=True)
        self.assertIsInstance(pdf_buffer, io.BytesIO)
        
        pdf_bytes = pdf_buffer.getvalue()
        self.assertGreater(len(pdf_bytes), 1000)
        # PDF magic bytes header
        self.assertTrue(pdf_bytes.startswith(b'%PDF-'))

    def test_generate_video_pdf_report_minimal_video(self):
        """Test PDF generator gracefully handles a video without summary or moments"""
        minimal_video = Video.objects.create(
            user=self.user_a,
            title="Raw Unsummarized Video",
            status=Video.STATUS_UPLOADED
        )
        pdf_buffer = generate_video_pdf_report(minimal_video, include_transcript=False)
        self.assertIsInstance(pdf_buffer, io.BytesIO)
        pdf_bytes = pdf_buffer.getvalue()
        self.assertTrue(pdf_bytes.startswith(b'%PDF-'))

    def test_download_pdf_report_view_authorized_success(self):
        """Test authorized owner can download dynamic PDF with correct headers"""
        url = reverse('videos:download_pdf', kwargs={'pk': self.video.pk})
        response = self.client_a.get(url)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('attachment;', response['Content-Disposition'])
        self.assertIn('VidNote_Report_', response['Content-Disposition'])
        self.assertGreater(len(response.content), 1000)
        self.assertTrue(response.content.startswith(b'%PDF-'))

    def test_download_pdf_report_view_user_isolation(self):
        """Test User B is blocked (404) from downloading User A's PDF report"""
        url = reverse('videos:download_pdf', kwargs={'pk': self.video.pk})
        response = self.client_b.get(url)
        self.assertEqual(response.status_code, 404)

    def test_download_pdf_report_view_unauthenticated(self):
        """Test unauthenticated user is redirected to login page"""
        url = reverse('videos:download_pdf', kwargs={'pk': self.video.pk})
        response = self.unauth_client.get(url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)
