from django.shortcuts import render

def home(request):
    """
    Home page view for VidNote AI.
    """
    context = {
        'title': 'VidNote AI - Intelligent Video Summarization & Notes',
        'project_name': 'VidNote AI',
        'project_tagline': 'Transform Videos into Structured Notes & Interactive RAG Knowledge Base',
    }
    return render(request, 'core/home.html', context)
