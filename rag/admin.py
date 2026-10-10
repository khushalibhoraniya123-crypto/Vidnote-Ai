from django.contrib import admin
from .models import ChatMessage


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('video', 'user', 'role', 'content_snippet', 'has_sources', 'created_at')
    list_filter = ('role', 'video__user', 'created_at')
    search_fields = ('content', 'video__title', 'user__username')
    readonly_fields = ('created_at',)
    ordering = ('-created_at',)

    def content_snippet(self, obj):
        return obj.content[:60] + ('...' if len(obj.content) > 60 else '')
    content_snippet.short_description = 'Message'
