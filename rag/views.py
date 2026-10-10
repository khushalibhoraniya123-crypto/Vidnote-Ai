from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from videos.models import Video
from .models import ChatMessage
from .services import answer_video_question, EmptyQueryError, VideoNotTranscribedError, RAGServiceError


@login_required
def video_chat_view(request, pk):
    """
    Dedicated AI Chatbot interface for a specific video.
    Allows students/users to converse with the video's transcript with grounded answers & source timestamps.
    Strictly user-permission isolated.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json'
        
        question = request.POST.get('question', '').strip()
        if not question and request.body:
            import json
            try:
                data = json.loads(request.body.decode('utf-8'))
                question = data.get('question', '').strip()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if not question:
            if is_ajax:
                return JsonResponse({'success': False, 'error': 'Please enter a valid question.'}, status=400)
            messages.error(request, "Please enter a question.")
            return redirect('rag:video_chat', pk=video.pk)

        try:
            assistant_msg = answer_video_question(
                video_instance=video,
                user=request.user,
                question_text=question
            )

            if is_ajax:
                return JsonResponse({
                    'success': True,
                    'user_message': {
                        'content': question,
                        'created_at': 'Just now'
                    },
                    'assistant_message': {
                        'content': assistant_msg.content,
                        'sources': assistant_msg.sources,
                        'has_sources': assistant_msg.has_sources,
                        'created_at': 'Just now'
                    }
                })

            messages.success(request, "Answer generated!")
        except (EmptyQueryError, VideoNotTranscribedError, RAGServiceError) as e:
            if is_ajax:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)
            messages.warning(request, str(e))
        except Exception as e:
            if is_ajax:
                return JsonResponse({'success': False, 'error': f"Failed to generate answer: {str(e)}"}, status=500)
            messages.error(request, f"Error generating answer: {str(e)}")

        return redirect('rag:video_chat', pk=video.pk)

    chat_messages = ChatMessage.objects.filter(video=video, user=request.user).order_by('created_at')

    context = {
        'video': video,
        'chat_messages': chat_messages,
        'has_messages': chat_messages.exists(),
        'messages_count': chat_messages.count(),
    }
    return render(request, 'rag/chat.html', context)


@login_required
def clear_chat_history_view(request, pk):
    """
    Clear chat history for a specific video. Strictly user-permission checked.
    """
    video = get_object_or_404(Video, pk=pk, user=request.user)

    if request.method == 'POST':
        deleted_count, _ = ChatMessage.objects.filter(video=video, user=request.user).delete()
        messages.success(request, f"Cleared {deleted_count} messages from chat history.")

    return redirect('rag:video_chat', pk=video.pk)
