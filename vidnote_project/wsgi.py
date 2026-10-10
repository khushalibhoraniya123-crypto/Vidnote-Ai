"""
WSGI config for vidnote_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vidnote_project.settings')

application = get_wsgi_application()
app = application

# Run migrations automatically on Vercel serverless cold start
if os.getenv('VERCEL'):
    try:
        from django.core.management import call_command
        call_command('migrate', interactive=False, verbosity=0)
    except Exception as _mig_err:
        pass




