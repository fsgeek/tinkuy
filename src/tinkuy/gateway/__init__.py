"""Gateway subpackage — HTTP gateway, server, stream handling, harness."""
from tinkuy.gateway._gateway import *  # noqa: F401,F403
from tinkuy.gateway._gateway import (  # noqa: F401 — private names used by tests
    _extract_user_content,
    _extract_response_content_from_json,
)
