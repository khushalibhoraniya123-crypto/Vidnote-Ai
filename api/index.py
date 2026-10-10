import os
import sys
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

django_app = None
init_error = None

try:
    from django.core.wsgi import get_wsgi_application
    django_app = get_wsgi_application()
except Exception:
    init_error = traceback.format_exc()

def app(environ, start_response):
    global django_app, init_error
    if init_error:
        body = f"<html><body><h1>VidNote AI - Startup Error</h1><pre>{init_error}</pre></body></html>".encode('utf-8')
        start_response('500 Internal Server Error', [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('Content-Length', str(len(body)))
        ])
        return [body]

    try:
        return django_app(environ, start_response)
    except Exception:
        err_body = f"<html><body><h1>VidNote AI - Runtime Error</h1><pre>{traceback.format_exc()}</pre></body></html>".encode('utf-8')
        start_response('500 Internal Server Error', [
            ('Content-Type', 'text/html; charset=utf-8'),
            ('Content-Length', str(len(err_body)))
        ])
        return [err_body]


