import os
import sys
from pathlib import Path

# Add workspace directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

try:
    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()
    app = application
except BaseException as exc:
    import traceback
    err_trace = traceback.format_exc()
    print(f"[FATAL DJANGO STARTUP ERROR] {err_trace}", file=sys.stderr)

    def app(environ, start_response):
        status = '500 Internal Server Error'
        headers = [('Content-Type', 'text/plain; charset=utf-8')]
        start_response(status, headers)
        return [f"Django Startup Error on Vercel:\n\n{err_trace}".encode('utf-8')]

# Expose handler for BaseHTTPRequestHandler compatibility
handler = app
