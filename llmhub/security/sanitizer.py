"""Input Sanitization & Request Validation for LLMHub.

Protects runtime execution against:
- Payload memory exhaustion via maximum request body limits
- C string buffer corruption via null byte injection (\x00) and dangerous control codes
- Malformed OpenAI chat message structures
- Prompt lengths exceeding maximum supported context windows
"""

import re
from typing import Any, Dict, List, Optional, Set, Union


class PayloadTooLargeError(Exception):
    """Raised when request body exceeds configured maximum byte limit."""
    pass


class ValidationError(Exception):
    """Raised when input parameters fail structural or validation rules."""
    pass


class InputSanitizer:
    """Sanitizes text inputs and validates request constraints."""

    # Disallowed control characters: ASCII 0-31 except \t (9), \n (10), \r (13), and ASCII 127 (DEL)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    VALID_ROLES: Set[str] = {"system", "user", "assistant"}

    @classmethod
    def sanitize_text(cls, text: Optional[str]) -> str:
        """Strip null bytes and non-printable control characters while preserving formatting whitespace."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Remove null bytes immediately
        clean = text.replace("\x00", "")
        # Remove dangerous control chars
        clean = cls._CONTROL_CHAR_RE.sub("", clean)
        return clean

    @classmethod
    def validate_request_size(cls, content_length: Optional[int], max_bytes: int) -> None:
        """Validate content-length header against maximum allowed body size."""
        if content_length is not None and content_length > max_bytes:
            raise PayloadTooLargeError(
                f"Request body size {content_length} bytes exceeds maximum allowed limit of {max_bytes} bytes."
            )

    @classmethod
    def validate_chat_messages(cls, messages: Any) -> List[Dict[str, str]]:
        """Validate list of chat messages, enforcing structure and roles."""
        if not isinstance(messages, list):
            raise ValidationError("Field 'messages' must be a list of chat message objects.")
        if len(messages) == 0:
            raise ValidationError("Field 'messages' must contain at least one message.")

        sanitized: List[Dict[str, str]] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                role = getattr(msg, "role")
                content = getattr(msg, "content", "")
            else:
                raise ValidationError(f"Message at index {i} is not a valid message object.")

            if not isinstance(role, str) or role.lower() not in cls.VALID_ROLES:
                raise ValidationError(
                    f"Message at index {i} has invalid role '{role}'. Allowed roles: {sorted(cls.VALID_ROLES)}"
                )

            sanitized_content = cls.sanitize_text(content)
            sanitized.append({"role": role.lower(), "content": sanitized_content})

        return sanitized

    @classmethod
    def validate_prompt_bounds(
        cls, prompt: str, max_ctx: int, max_allowed_chars: Optional[int] = None
    ) -> None:
        """Heuristic check ensuring prompt text does not grossly exceed context length bounds.
        
        Assumes average token length of ~2-4 chars. If prompt characters exceed
        max_ctx * 8, it is guaranteed to exceed max_ctx tokens.
        """
        limit = max_allowed_chars or (max_ctx * 8)
        if len(prompt) > limit:
            raise ValidationError(
                f"Prompt length ({len(prompt)} characters) exceeds maximum supported context limit "
                f"for model context window of {max_ctx} tokens (max {limit} chars)."
            )
