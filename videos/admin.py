from django.contrib import admin
from .models import Video, TranscriptSegment, Summary, ImportantMoment, ActionItem

class TranscriptSegmentInline(admin.TabularInline):
    model = TranscriptSegment
    extra = 0
    fields = ('segment_order', 'start_time', 'end_time', 'text')
    readonly_fields = ('segment_order', 'start_time', 'end_time', 'text')
    can_delete = False

class ImportantMomentInline(admin.TabularInline):
    model = ImportantMoment
    extra = 0
    fields = ('title', 'start_time', 'end_time', 'importance', 'description')
    readonly_fields = ('created_at',)

class ActionItemInline(admin.TabularInline):
    model = ActionItem
    extra = 0
    fields = ('task', 'priority', 'status', 'timestamp', 'description')
    readonly_fields = ('created_at',)

@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'status', 'language', 'formatted_duration', 'formatted_file_size', 'created_at')
    list_filter = ('status', 'language', 'created_at')
    search_fields = ('title', 'description', 'user__username', 'original_filename')
    readonly_fields = ('original_filename', 'file_size', 'duration', 'created_at', 'updated_at')
    inlines = [TranscriptSegmentInline, ImportantMomentInline, ActionItemInline]
    list_per_page = 25

@admin.register(TranscriptSegment)
class TranscriptSegmentAdmin(admin.ModelAdmin):
    list_display = ('video', 'segment_order', 'formatted_start_time', 'formatted_end_time', 'text')
    list_filter = ('video__user', 'video')
    search_fields = ('text', 'video__title')
    ordering = ('video', 'segment_order')

@admin.register(Summary)
class SummaryAdmin(admin.ModelAdmin):
    list_display = ('video', 'model_used', 'created_at', 'updated_at')
    search_fields = ('video__title', 'short_summary', 'detailed_summary')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(ImportantMoment)
class ImportantMomentAdmin(admin.ModelAdmin):
    list_display = ('video', 'title', 'formatted_start_time', 'formatted_end_time', 'importance', 'created_at')
    list_filter = ('importance', 'video__user', 'video')
    search_fields = ('title', 'description', 'video__title')
    ordering = ('video', 'start_time')

@admin.register(ActionItem)
class ActionItemAdmin(admin.ModelAdmin):
    list_display = ('video', 'task', 'priority', 'status', 'formatted_timestamp', 'created_at')
    list_filter = ('status', 'priority', 'video__user', 'video')
    search_fields = ('task', 'description', 'video__title')
    ordering = ('video', 'status', '-created_at')

