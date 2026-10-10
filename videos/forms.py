import os
from django import forms
from .models import Video

# Allowed video extensions
ALLOWED_EXTENSIONS = ['.mp4', '.mkv', '.mov', '.avi', '.webm']
# Maximum allowed file size in bytes (100 MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

class VideoUploadForm(forms.ModelForm):
    """
    Form for uploading video files with type and size validation.
    """
    class Meta:
        model = Video
        fields = ['title', 'description', 'video_file']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control form-control-lg',
                'placeholder': 'e.g., CS101 - Lecture 04: Operating Systems'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Optional notes or summary topic for this video...'
            }),
            'video_file': forms.ClearableFileInput(attrs={
                'class': 'form-control',
                'accept': 'video/mp4,video/mkv,video/quicktime,video/x-msvideo,video/webm,.mp4,.mkv,.mov,.avi,.webm'
            }),
        }

    def clean_video_file(self):
        video_file = self.cleaned_data.get('video_file')
        if not video_file:
            raise forms.ValidationError("Please select a video file to upload.")

        # 1. Validate file extension
        ext = os.path.splitext(video_file.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            allowed_str = ", ".join(ALLOWED_EXTENSIONS)
            raise forms.ValidationError(
                f"Unsupported file format '{ext}'. Allowed formats are: {allowed_str}"
            )

        # 2. Validate empty file
        if video_file.size == 0:
            raise forms.ValidationError("The uploaded video file is empty (0 bytes). Please upload a valid video.")

        # 3. Validate file size (max 100MB)
        if video_file.size > MAX_UPLOAD_SIZE:
            max_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
            current_mb = video_file.size / (1024 * 1024)
            raise forms.ValidationError(
                f"File size exceeds the {max_mb} MB limit (Uploaded file is {current_mb:.1f} MB)."
            )

        return video_file
