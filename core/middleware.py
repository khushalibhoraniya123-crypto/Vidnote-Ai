"""
Middleware to normalize Vercel serverless request paths.
Strips internal rewrite artifacts such as /api/index.py from PATH_INFO.
"""

class VercelPathNormalizationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info
        if path.startswith('/api/index.py'):
            request.path_info = path[len('/api/index.py'):] or '/'
        elif path.startswith('/api/index'):
            request.path_info = path[len('/api/index'):] or '/'
        elif path.startswith('/api'):
            request.path_info = path[len('/api'):] or '/'
        return self.get_response(request)
