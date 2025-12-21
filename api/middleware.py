"""
Custom middleware for API enhancements.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class VersionHeaderMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add version headers to all responses.

    Adds:
    - X-API-Version: API version number
    - X-Model-Version: Model identifier
    """

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add version headers to response.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            Response with version headers added
        """
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Model-Version"] = "resnet50-fold0"
        return response
