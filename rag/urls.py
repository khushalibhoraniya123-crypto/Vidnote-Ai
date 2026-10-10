from django.urls import path
from . import views

app_name = 'rag'

urlpatterns = [
    path('<int:pk>/', views.video_chat_view, name='video_chat'),
    path('<int:pk>/chat/', views.video_chat_view, name='chat'),
    path('<int:pk>/clear/', views.clear_chat_history_view, name='clear_chat'),
]
