"""
Unit and Integration Tests for VidNote AI RAG App.
Validates chunking, embeddings generation, ChromaDB vector indexing,
metadata retention, semantic similarity search, and video-level isolation.
"""

import os
import shutil
import tempfile
from django.test import TestCase, override_settings
from django.contrib.auth.models import User
from videos.models import Video, TranscriptSegment
from rag.models import ChatMessage
from rag.embeddings import generate_embedding, generate_embeddings, generate_fallback_embedding
from rag.vector_store import (
    get_or_create_collection,
    add_chunks_to_vector_store,
    delete_video_chunks,
    query_vector_store
)
from rag.retrieval import chunk_transcript_segments, search_transcript, EmptyQueryError
from rag.services import (
    index_video_transcript,
    semantic_search_video,
    answer_video_question,
    NOT_FOUND_MESSAGE,
    VideoNotTranscribedError
)
from django.test import Client
from django.urls import reverse


class RAGPipelineTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_chroma_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_chroma_dir, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        self.user_a = User.objects.create_user(username=f'raguser_a_{self._testMethodName}', password='Password123!')
        self.user_b = User.objects.create_user(username=f'raguser_b_{self._testMethodName}', password='Password123!')
        
        self.client_a = Client()
        self.client_a.login(username=f'raguser_a_{self._testMethodName}', password='Password123!')

        self.client_b = Client()
        self.client_b.login(username=f'raguser_b_{self._testMethodName}', password='Password123!')

        self.video_a = Video.objects.create(
            user=self.user_a,
            title="Data Structures and Algorithms"
        )
        self.video_b = Video.objects.create(
            user=self.user_a,
            title="Cell Biology and Genetics"
        )
        delete_video_chunks(self.video_a.pk)
        delete_video_chunks(self.video_b.pk)

    def tearDown(self):
        if hasattr(self, 'video_a') and self.video_a.pk:
            delete_video_chunks(self.video_a.pk)
        if hasattr(self, 'video_b') and self.video_b.pk:
            delete_video_chunks(self.video_b.pk)

    def test_embeddings_generation_and_normalization(self):
        """Test dense vector generation for single strings and batches"""
        text = "Deep learning neural networks and gradient descent optimization"
        vec = generate_embedding(text)
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

        batch = ["First segment of speech", "Second segment of speech"]
        vecs = generate_embeddings(batch)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), len(vecs[1]))

    def test_chunk_transcript_segments_preserves_metadata(self):
        """Test transcript chunking maintains exact start and end timestamps"""
        seg1 = TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=10.0,
            text="Welcome to algorithms class. Today we introduce sorting techniques.",
            segment_order=0
        )
        seg2 = TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=10.0,
            end_time=25.0,
            text="Bubble sort repeatedly steps through the list to swap adjacent elements.",
            segment_order=1
        )
        seg3 = TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=25.0,
            end_time=45.0,
            text="Next we look at quicksort and divide and conquer pivot partitioning.",
            segment_order=2
        )

        segments = [seg1, seg2, seg3]
        chunks = chunk_transcript_segments(segments, max_chunk_words=15, overlap_words=0)
        
        self.assertGreater(len(chunks), 0)
        for chk in chunks:
            self.assertIn('start_time', chk)
            self.assertIn('end_time', chk)
            self.assertIn('text', chk)
            self.assertIn('chunk_id', chk)
            self.assertGreaterEqual(chk['end_time'], chk['start_time'])

    def test_semantic_search_multiple_questions_ranking(self):
        """Test semantic similarity search retrieves and ranks relevant timestamps across multiple questions"""
        # Populate lecture with distinct topical segments
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=15.0,
            text="Introduction to simple quadratic sorting including bubble sort and insertion sort algorithms.",
            segment_order=0
        )
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=16.0,
            end_time=35.0,
            text="Quicksort uses divide and conquer with a chosen pivot element to partition subarrays efficiently.",
            segment_order=1
        )
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=36.0,
            end_time=55.0,
            text="Merge sort guarantees O of N log N asymptotic time complexity using recursive split and merge steps.",
            segment_order=2
        )
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=56.0,
            end_time=80.0,
            text="Breadth first search BFS and depth first search DFS explore vertices and find shortest paths in graphs.",
            segment_order=3
        )

        # Index the video
        idx_result = index_video_transcript(self.video_a)
        self.assertEqual(idx_result['status'], 'indexed')
        self.assertGreater(idx_result['chunks_count'], 0)

        # Question 1: Quick Sort Pivot Partitioning
        q1_results = semantic_search_video(self.video_a, "How does pivot partitioning work in divide and conquer?", top_k=2)
        self.assertGreater(len(q1_results), 0)
        top_match_1 = q1_results[0]
        self.assertIn('formatted_start_time', top_match_1)
        self.assertIn('Quicksort', top_match_1['text'])

        # Question 2: Merge Sort Time Complexity
        q2_results = semantic_search_video(self.video_a, "What is the time complexity and recursion for merge sort?", top_k=2)
        self.assertGreater(len(q2_results), 0)
        top_match_2 = q2_results[0]
        self.assertIn('Merge sort', top_match_2['text'])

        # Question 3: Graph Shortest Path Search
        q3_results = semantic_search_video(self.video_a, "How do you traverse graphs and find shortest paths with BFS?", top_k=2)
        self.assertGreater(len(q3_results), 0)
        top_match_3 = q3_results[0]
        self.assertIn('Breadth first search', top_match_3['text'])

    def test_video_level_isolation_filtering(self):
        """Test semantic search on Video A never leaks chunks belonging to Video B"""
        # Video A: Physics
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=20.0,
            text="Newton second law relates force mass and acceleration according to F equals m a.",
            segment_order=0
        )
        # Video B: Biology
        TranscriptSegment.objects.create(
            video=self.video_b,
            start_time=0.0,
            end_time=20.0,
            text="Photosynthesis converts solar light photons into chemical glucose in chloroplasts.",
            segment_order=0
        )

        index_video_transcript(self.video_a)
        index_video_transcript(self.video_b)

        # Search Video A for biology concept -> should only return Video A chunks, never Video B
        results = semantic_search_video(self.video_a, "Where does photosynthesis occur in chloroplasts?", top_k=3)
        for r in results:
            self.assertEqual(r['video_id'], self.video_a.pk)
            self.assertNotIn("chloroplasts", r['text'].lower())

    def test_repeatable_indexing_without_duplicates(self):
        """Test re-indexing replaces existing vector chunks cleanly without duplicate entries"""
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=15.0,
            text="Initial lecture on computer networking protocols and TCP handshake.",
            segment_order=0
        )

        # First indexing
        res1 = index_video_transcript(self.video_a)
        count1 = res1['chunks_count']

        # Second indexing
        res2 = index_video_transcript(self.video_a)
        count2 = res2['chunks_count']

        self.assertEqual(count1, count2)

        # Query search returns clean result count without duplicates
        results = semantic_search_video(self.video_a, "TCP handshake networking", top_k=10)
        chunk_ids = [r['chunk_id'] for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_error_handling_empty_query_and_missing_transcript(self):
        """Test empty query raises EmptyQueryError and empty video raises VideoNotTranscribedError"""
        # Empty query
        with self.assertRaises(EmptyQueryError):
            semantic_search_video(self.video_a, "   ")

        # Video with no transcript
        empty_video = Video.objects.create(user=self.user_a, title="Empty Video")
        with self.assertRaises(VideoNotTranscribedError):
            semantic_search_video(empty_video, "What is machine learning?")

    # =========================================================================
    # STEP 10: RAG CHATBOT TESTS
    # =========================================================================

    def test_chat_message_model(self):
        """Test ChatMessage creation, roles, and helper properties"""
        msg = ChatMessage.objects.create(
            video=self.video_a,
            user=self.user_a,
            role=ChatMessage.ROLE_USER,
            content="What is time complexity of quicksort?"
        )
        self.assertTrue(msg.is_user)
        self.assertFalse(msg.is_assistant)
        self.assertFalse(msg.has_sources)

        asst_msg = ChatMessage.objects.create(
            video=self.video_a,
            user=self.user_a,
            role=ChatMessage.ROLE_ASSISTANT,
            content="Average time complexity is O(N log N).",
            sources=[{'start_time': 12.0, 'formatted_start_time': '00:00:12'}]
        )
        self.assertTrue(asst_msg.is_assistant)
        self.assertTrue(asst_msg.has_sources)

    def test_answer_video_question_grounded_and_stores_messages(self):
        """Test answer_video_question stores conversation history and provides source timestamps"""
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=10.0,
            end_time=30.0,
            text="Binary search trees BST have logarithmic lookup time when balanced.",
            segment_order=0
        )
        index_video_transcript(self.video_a)

        assistant_msg = answer_video_question(
            video_instance=self.video_a,
            user=self.user_a,
            question_text="What are the properties of balanced binary search trees?"
        )

        self.assertIsNotNone(assistant_msg)
        self.assertEqual(assistant_msg.role, ChatMessage.ROLE_ASSISTANT)
        self.assertIn("binary search", assistant_msg.content.lower())
        self.assertEqual(ChatMessage.objects.filter(video=self.video_a).count(), 2)

    def test_answer_video_question_hallucination_guard_when_info_missing(self):
        """Test when question asks about unavailable facts, the AI responds with NOT_FOUND_MESSAGE"""
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=15.0,
            text="Welcome to basic Python programming. We cover lists, tuples, and loops.",
            segment_order=0
        )
        index_video_transcript(self.video_a)

        # Ask about quantum gravity (completely absent from transcript)
        assistant_msg = answer_video_question(
            video_instance=self.video_a,
            user=self.user_a,
            question_text="How does string theory reconcile general relativity with quantum mechanics?"
        )

        self.assertEqual(assistant_msg.content, NOT_FOUND_MESSAGE)
        self.assertEqual(len(assistant_msg.sources), 0)

    def test_video_chat_view_renders_and_handles_post(self):
        """Test video_chat_view GET and POST requests"""
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=20.0,
            text="Operating systems manage hardware resources, process scheduling, and memory allocation.",
            segment_order=0
        )
        index_video_transcript(self.video_a)

        # GET request
        url = reverse('rag:video_chat', kwargs={'pk': self.video_a.pk})
        res = self.client_a.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "VidNote AI Chatbot")

        # POST form request
        post_res = self.client_a.post(url, {'question': 'What does an operating system manage?'}, follow=True)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(ChatMessage.objects.filter(video=self.video_a).count(), 2)

    def test_video_chat_view_ajax_json(self):
        """Test video_chat_view AJAX endpoint returning JSON payload"""
        TranscriptSegment.objects.create(
            video=self.video_a,
            start_time=0.0,
            end_time=20.0,
            text="Recursion is a method where a function solves a problem by calling itself.",
            segment_order=0
        )
        index_video_transcript(self.video_a)

        url = reverse('rag:video_chat', kwargs={'pk': self.video_a.pk})
        import json
        ajax_res = self.client_a.post(
            url,
            data=json.dumps({'question': 'What is recursion?'}),
            content_type='application/json',
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(ajax_res.status_code, 200)
        data = ajax_res.json()
        self.assertTrue(data['success'])
        self.assertIn('assistant_message', data)
        self.assertIn('content', data['assistant_message'])

    def test_clear_chat_history_view(self):
        """Test clearing chat history for a video"""
        ChatMessage.objects.create(
            video=self.video_a,
            user=self.user_a,
            role=ChatMessage.ROLE_USER,
            content="Hello"
        )
        self.assertEqual(ChatMessage.objects.filter(video=self.video_a).count(), 1)

        clear_url = reverse('rag:clear_chat', kwargs={'pk': self.video_a.pk})
        res = self.client_a.post(clear_url, follow=True)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(ChatMessage.objects.filter(video=self.video_a).count(), 0)

    def test_chat_user_isolation_permissions(self):
        """User B cannot access, ask questions in, or clear User A's video chat"""
        # User B cannot access User A's chat view (404)
        chat_url = reverse('rag:video_chat', kwargs={'pk': self.video_a.pk})
        res_b = self.client_b.get(chat_url)
        self.assertEqual(res_b.status_code, 404)

        # User B cannot POST question to User A's video chat (404)
        post_res_b = self.client_b.post(chat_url, {'question': 'Hacked question?'})
        self.assertEqual(post_res_b.status_code, 404)

        # User B cannot clear User A's chat history (404)
        clear_url = reverse('rag:clear_chat', kwargs={'pk': self.video_a.pk})
        clear_res_b = self.client_b.post(clear_url)
        self.assertEqual(clear_res_b.status_code, 404)

