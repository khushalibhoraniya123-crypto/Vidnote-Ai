from django.db import models
from django.contrib.auth.models import User
from videos.models import Video


class ChatMessage(models.Model):
    """
    Model representing a question-answer conversation turn grounded in a specific video transcript.
    """
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'

    ROLE_CHOICES = [
        (ROLE_USER, 'User'),
        (ROLE_ASSISTANT, 'AI Assistant'),
    ]

    video = models.ForeignKey(
        Video,
        on_delete=models.CASCADE,
        related_name='chat_messages',
        help_text="The video this chat message is grounded in"
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='rag_chat_messages',
        help_text="The user who asked or received this message"
    )
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default=ROLE_USER,
        db_index=True
    )
    content = models.TextField(help_text="Message content (user query or grounded AI answer)")
    sources = models.JSONField(
        default=list,
        blank=True,
        help_text="List of referenced transcript chunks with timestamps and snippet text"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'Chat Message'
        verbose_name_plural = 'Chat Messages'

    def __str__(self):
        return f"[{self.video.title}] {self.get_role_display()}: {self.content[:40]}..."

    @property
    def is_user(self) -> bool:
        return self.role == self.ROLE_USER

    @property
    def is_assistant(self) -> bool:
        return self.role == self.ROLE_ASSISTANT

    @property
    def has_sources(self) -> bool:
        return bool(self.sources and len(self.sources) > 0)
