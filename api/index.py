import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

# Auto-migrate SQLite in /tmp for serverless instances
if os.getenv('VERCEL', '0') == '1' or bool(os.getenv('VERCEL_URL')):
    try:
        from django.core.management import call_command
        db_path = Path('/tmp') / 'db.sqlite3'
        if not db_path.exists():
            call_command('migrate', interactive=False)
    except Exception as exc:
        print(f"Vercel DB setup notice: {exc}")

from django.core.wsgi import get_wsgi_application

app = get_wsgi_application()

