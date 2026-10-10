import os
import sys
from pathlib import Path

# Add project root directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

from django.core.wsgi import get_wsgi_application

django_app = get_wsgi_application()

def app(environ, start_response):
    # Normalize PATH_INFO to counteract Vercel internal rewrite destination distortion
    path = environ.get('PATH_INFO', '')
    if path.startswith('/api/index.py'):
        environ['PATH_INFO'] = path[len('/api/index.py'):] or '/'
    elif path.startswith('/api/index'):
        environ['PATH_INFO'] = path[len('/api/index'):] or '/'
    elif path.startswith('/api'):
        environ['PATH_INFO'] = path[len('/api'):] or '/'
    
    return django_app(environ, start_response)

application = app
handler = app
