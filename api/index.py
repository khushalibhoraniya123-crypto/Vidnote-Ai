import os
import sys
import traceback
from pathlib import Path

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

try:
    from django.core.wsgi import get_wsgi_application
    _django_app = get_wsgi_application()

    def app(environ, start_response):
        try:
            return _django_app(environ, start_response)
        except Exception:
            err = traceback.format_exc()
            status = '500 Internal Server Error'
            headers = [('Content-Type', 'text/html; charset=utf-8')]
            start_response(status, headers)
            return [f"<!DOCTYPE html><html><head><title>VidNote AI - Error</title></head><body style='font-family:sans-serif;padding:30px;'><h2>Django Runtime Error on Vercel</h2><pre style='background:#f1f5f9;padding:15px;border-radius:6px;overflow:auto;'>{err}</pre></body></html>".encode('utf-8')]

except Exception:
    startup_err = traceback.format_exc()

    def app(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'text/html; charset=utf-8')]
        start_response(status, headers)
        return [f"<!DOCTYPE html><html><head><title>VidNote AI - Startup Error</title></head><body style='font-family:sans-serif;padding:30px;'><h2>Django Startup Error on Vercel</h2><pre style='background:#f1f5f9;padding:15px;border-radius:6px;overflow:auto;'>{startup_err}</pre></body></html>".encode('utf-8')]
