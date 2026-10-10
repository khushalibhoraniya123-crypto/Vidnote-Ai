from django.urls import path
from . import views

app_name = 'videos'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('<int:pk>/', views.detail_view, name='detail'),
    path('<int:pk>/summary/', views.summary_view, name='summary'),
    path('<int:pk>/generate-summary/', views.generate_summary_action_view, name='generate_summary'),
    path('<int:pk>/action-items/', views.action_items_view, name='action_items'),
    path('<int:pk>/extract-action-items/', views.extract_action_items_action_view, name='extract_action_items'),
    path('<int:pk>/action-items/<int:item_id>/toggle/', views.toggle_action_item_status_view, name='toggle_action_item_status'),
    path('<int:pk>/action-items/<int:item_id>/priority/', views.update_action_item_priority_view, name='update_action_item_priority'),
    path('<int:pk>/action-items/<int:item_id>/delete/', views.delete_action_item_view, name='delete_action_item'),
    path('<int:pk>/action-items/create/', views.create_action_item_view, name='create_action_item'),
    path('<int:pk>/detect-moments/', views.detect_moments_action_view, name='detect_moments'),
    path('<int:pk>/pdf/', views.download_pdf_report_view, name='download_pdf'),
    path('<int:pk>/extract-audio/', views.extract_audio_action_view, name='extract_audio'),
    path('<int:pk>/transcribe/', views.transcribe_action_view, name='transcribe_action'),
    path('<int:pk>/process/', views.process_pipeline_view, name='process_pipeline'),
    path('<int:pk>/status/', views.processing_status_api_view, name='processing_status'),
    path('<int:pk>/delete/', views.delete_view, name='delete'),
]
