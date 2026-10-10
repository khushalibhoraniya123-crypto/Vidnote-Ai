import os
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        if os.getenv('VERCEL'):
            import threading
            def _auto_migrate():
                try:
                    from django.core.management import call_command
                    call_command('migrate', interactive=False, verbosity=0)
                except Exception:
                    pass
            threading.Thread(target=_auto_migrate, daemon=True).start()
