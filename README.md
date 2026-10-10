# 🎥 VidNote AI
### *Intelligent Video Summarization, Smart Notes, and RAG-Based Question Answering System*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1+-092E20.svg)](https://www.djangoproject.com/)
[![Whisper](https://img.shields.io/badge/ASR-Faster--Whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence--Transformers-green.svg)](https://www.sbert.net/)
[![ReportLab](https://img.shields.io/badge/PDF-ReportLab-red.svg)](https://www.reportlab.com/)

---

## 📌 1. Project Overview & Description

**VidNote AI** is an advanced end-to-end multi-modal video intelligence platform designed for students, educators, researchers, and professionals. It automatically ingests educational, meeting, or lecture videos, extracts audio, generates timestamped speech-to-text transcripts using local Faster-Whisper models, synthesizes multi-tier AI summaries, detects important moments/timestamps, extracts actionable tasks with priority assignments, indexes content into a ChromaDB vector store, powers a Retrieval-Augmented Generation (RAG) conversational assistant with clickable video timestamp citations, and generates downloadable PDF reports.

---

## 🛑 2. Problem Statement

With the exponential surge in recorded online lectures, technical webinars, meeting recordings, and tutorials:
- **Time Consumption**: Watching multi-hour videos to find specific concepts or action items is inefficient.
- **Lack of Searchability**: Video content is fundamentally opaque; searching for specific topics requires tedious scrubbing.
- **Information Retention**: Manual note-taking is time-consuming and often misses critical details, deadlines, and key moments.
- **Context Loss in AI**: Standard AI models hallucinate when asked about video contents without ground-truth context.

**VidNote AI solves this** by transforming unstructured video data into structured, interactive, searchable, and actionable knowledge bases.

---

## ✨ 3. Core Features

1. **Secure User Authentication & Multi-Tenancy**:
   - Django auth with custom profile management, CSRF-protected forms, and complete cross-user data isolation.
2. **Video Ingestion & Validation**:
   - Multi-format support (`.mp4`, `.mov`, `.avi`, `.mkv`), 0-byte upload rejection, and 100MB size limits.
3. **Automated Audio Extraction**:
   - Headless audio stream demuxing via `imageio-ffmpeg` to 16kHz mono WAV format.
4. **Fast Speech-to-Text Transcription**:
   - `faster-whisper` CTranslate2 implementation with sentence-level timestamps and language detection.
5. **Multi-Tier AI Summarization**:
   - Executive Brief, Detailed Overview, and Bulleted Key Takeaways with local heuristic fallback engines.
6. **Key Timestamp & Moment Detection**:
   - Identifies critical conceptual breakthroughs, definitions, and topic transitions with "Jump to Timestamp" HTML5 video seeking.
7. **Action Item & Task Extraction**:
   - Automatically detects assignments, meetings, and deadlines with Priority levels (`High`, `Medium`, `Low`) and interactive status toggles (`Pending` / `Completed`).
8. **Retrieval-Augmented Generation (RAG)**:
   - Chunked semantic indexing using `sentence-transformers/all-MiniLM-L6-v2` and local persistent `ChromaDB`.
9. **Interactive Grounded AI Chatbot**:
   - Video-scoped conversational assistant that provides strictly grounded answers accompanied by clickable source timestamp links.
10. **Automated PDF Report Generation**:
    - Professional ReportLab-generated documentation with executive summaries, key moments, action item tables, and optional full transcripts.
11. **Fault-Tolerant Processing Pipeline**:
    - Step-by-step job tracking (`queued` → `processing` → `completed` / `failed`) with safe error containment and single-click retries.

---

## 💻 4. Technology Stack

- **Backend Framework**: Python 3.10+ / Django 5.1+
- **Database**: SQLite (Development) / PostgreSQL compatible
- **Speech Recognition (ASR)**: `faster-whisper` (CTranslate2 INT8 engine)
- **Audio Processing**: `imageio-ffmpeg`
- **Embedding Model**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database**: `ChromaDB` (Persistent local collection per video)
- **LLM Integrations**: Google Gemini (`gemini-2.5-flash`), OpenAI (`gpt-4o-mini`), Groq (`llama-3.3-70b-versatile`), and Rule-based Heuristic Fallback
- **PDF Engine**: `ReportLab`
- **Frontend / UI**: HTML5, CSS3, Bootstrap 5.3, Bootstrap Icons, Vanilla JavaScript

---

## 🏛️ 5. System Architecture

```
[ Video File (.mp4/.mov/.avi/.mkv) ]
                 │
                 ▼
      [ Stage 1: File Validation ]
                 │
                 ▼
      [ Stage 2: Audio Extraction (imageio-ffmpeg) ] ───► [ 16kHz Mono .wav ]
                 │
                 ▼
      [ Stage 3: Faster-Whisper ASR ] ──────────────► [ Timestamped Segments ]
                 │
                 ├─────────────────────────────────────────┬─────────────────────────────────────────┐
                 ▼                                         ▼                                         ▼
      [ Stage 4: AI Summary Engine ]             [ Stage 5: Key Moments ]                  [ Stage 6: Action Items ]
        - Executive Summary                        - Start/End Times                         - Tasks & Descriptions
        - Detailed Overview                        - Importance Levels                       - Priority (High/Med/Low)
        - Key Points                               - HTML5 Seek Triggers                     - Status Toggle
                 │                                         │                                         │
                 └─────────────────────────────────────────┼─────────────────────────────────────────┘
                                                           ▼
                                            [ Stage 7: ChromaDB RAG Indexing ]
                                              - Sentence-Transformers Embedding
                                              - Segment Chunk Metadata Storage
                                                           │
                                                           ▼
                                      [ Stage 8: User Interactions & Delivery ]
                                        - Interactive Video Dashboard
                                        - Grounded RAG Chat with Timestamp Jump
                                        - ReportLab Dynamic PDF Export
```

---

## 📂 6. Project Structure

```
VidNote AI/
├── accounts/                   # User authentication, registration, login/logout, profile
│   ├── models.py
│   ├── urls.py
│   └── views.py
├── core/                       # Core views (Landing page, about)
│   ├── urls.py
│   └── views.py
├── rag/                        # Retrieval-Augmented Generation module
│   ├── embeddings.py           # Sentence-transformers embedding wrapper
│   ├── vector_store.py         # ChromaDB client & collection manager
│   ├── retrieval.py            # Cosine similarity context retrieval
│   ├── services.py             # Grounded prompt construction & chat response generator
│   ├── models.py               # ChatMessage model
│   ├── urls.py
│   └── views.py
├── reports/                    # PDF Report generation module
│   ├── pdf.py                  # ReportLab canvas, flowables, and styled PDF builders
│   ├── urls.py
│   └── views.py
├── videos/                     # Video upload, pipeline, transcript, moments & tasks
│   ├── services/
│   │   ├── audio_extractor.py  # FFmpeg audio demuxing
│   │   ├── transcriber.py      # Faster-Whisper ASR
│   │   ├── summarizer.py       # Gemini/OpenAI/Groq/Fallback summarizer
│   │   ├── timestamps.py       # Important moments analyzer
│   │   ├── action_items.py     # Action item task extractor
│   │   └── pipeline.py         # 8-stage fault-tolerant orchestration pipeline
│   ├── models.py               # Video, ProcessingJob, Transcript, ImportantMoment, ActionItem
│   ├── forms.py                # Video upload validation forms
│   ├── urls.py
│   ├── views.py
│   └── tests.py                # 45 comprehensive unit & security integration tests
├── templates/                  # Reusable Bootstrap 5 templates
│   ├── base.html
│   ├── accounts/
│   ├── core/
│   ├── videos/
│   └── rag/
├── static/                     # CSS stylesheets, JS scripts, brand assets
├── media/                      # User uploaded videos and extracted audio (gitignored)
├── vidnote_project/            # Django project settings and root routing
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .env.example                # Template for environment variables
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Python dependencies
└── manage.py
```

---

## ⚙️ 7. Installation & Setup Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/vidnote-ai.git
cd "VidNote AI"
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 8. Environment Configuration (`.env`)

Copy `.env.example` to create your active `.env` file:
```bash
copy .env.example .env     # Windows
cp .env.example .env       # Linux / macOS
```

Configure the following variables in `.env`:
```ini
# Django Settings
SECRET_KEY=django-insecure-your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost

# Whisper Model Configuration
WHISPER_MODEL_SIZE=tiny
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# LLM Provider Configuration (gemini, openai, groq, or fallback)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=
GROQ_API_KEY=
LLM_MODEL_NAME=gemini-2.5-flash
```

> **Note**: If no API key is provided, VidNote AI automatically activates its **intelligent heuristic fallback engine**, enabling 100% offline demonstration without crashing!

---

## 🎬 9. FFmpeg Setup

VidNote AI uses `imageio-ffmpeg`, which **automatically bundles and manages standalone FFmpeg binaries**. No manual operating system FFmpeg installation or PATH configuration is required.

*(Optional)*: If you wish to specify a system binary, set `FFMPEG_PATH=C:\path\to\ffmpeg.exe` in your `.env`.

---

## 🗄️ 10. Database Migration & Initialization

Run migrations to create the database schema:
```bash
python manage.py migrate
```

Create a superuser (optional for admin panel access):
```bash
python manage.py createsuperuser
```

---

## 🚀 11. Running the Application

Start the development server:
```bash
python manage.py runserver
```
Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧠 12. How RAG (Retrieval-Augmented Generation) Works

1. **Chunking**: The video transcript segments are grouped into overlapping semantic chunks with metadata (`video_id`, `chunk_id`, `start_time`, `end_time`).
2. **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` encodes each chunk into a 384-dimensional dense vector.
3. **Vector Storage**: Vectors are stored in a video-isolated collection within `ChromaDB`.
4. **Query Retrieval**: When a user submits a question in the AI Chat, the question is embedded, and ChromaDB retrieves top-$K$ cosine similarity chunks matching that specific video.
5. **Grounded Generation**: The LLM is prompted with strict instruction to **answer ONLY using the provided transcript context**.
6. **Timestamp Attribution**: Each retrieved source chunk includes its starting timestamp, displayed as an interactive button that jumps the video player to that exact second.

---

## 🧪 13. Running Automated Tests

Run the full automated test suite (45 unit, service, pipeline, and security tests):
```bash
python manage.py test
```

---

## ⚠️ 14. Common Errors & Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Unsupported file format` | Uploading non-video file | Upload valid `.mp4`, `.mov`, `.avi`, or `.mkv` files. |
| `File size exceeds 100MB` | Video file too large | Use short clips (< 100MB) for demonstration. |
| `Gemini API quota exceeded` | Free tier rate limits | VidNote AI automatically falls back to heuristic extraction seamlessly. |
| `ChromaDB SQLite error` | Corrupt vector index | Delete `media/chroma_db/` and re-index the video. |

---

## 🔮 15. Future Scope

- Multi-language translation and dubbed audio transcription.
- Multi-video semantic search across an entire semester's course lectures.
- Visual OCR integration for slides, blackboard equations, and diagram recognition.
- Export integrations to Notion, Anki Flashcards, and Google Classroom.

---

## 📜 16. License

Developed for Academic Demonstration & Educational Purposes. Released under the **MIT License**.
