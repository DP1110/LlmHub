"""Jinja2 Chat Template Renderer for GGUF models.

Parses and applies Jinja2 chat templates embedded in GGUF metadata
(tokenizer.chat_template), with robust fallbacks for ChatML and Llama architectures.
"""

from typing import Any, Dict, List, Optional
import jinja2

# Standard fallback templates
FALLBACK_CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

FALLBACK_LLAMA3 = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

FALLBACK_BASIC = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "System: {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "Assistant: "
    "{% endif %}"
)

def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)

class ChatTemplateEngine:
    """Manages chat template compilation and rendering."""

    def __init__(
        self,
        template_str: Optional[str] = None,
        architecture: str = "llama",
        bos_token: str = "<|begin_of_text|>",
        eos_token: str = "<|end_of_text|>"
    ):
        self.architecture = architecture.lower()
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.template_source = self._select_template(template_str)
        self.env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False
        )
        # Register custom Jinja functions commonly used by HuggingFace templates
        self.env.globals["raise_exception"] = _raise_exception
        self.template = self.env.from_string(self.template_source)

    def _select_template(self, template_str: Optional[str]) -> str:
        if template_str and template_str.strip():
            return template_str.strip()
        
        if "qwen" in self.architecture:
            return FALLBACK_CHATML
        elif "llama" in self.architecture:
            return FALLBACK_LLAMA3
        else:
            return FALLBACK_BASIC

    def render(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **extra_vars: Any
    ) -> str:
        """Render conversation history into a model prompt string."""
        context = {
            "messages": messages,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "add_generation_prompt": add_generation_prompt,
            **extra_vars
        }
        try:
            return self.template.render(**context)
        except Exception:
            # Fallback to ChatML if complex custom template throws template errors
            fallback_tmpl = self.env.from_string(FALLBACK_CHATML)
            return fallback_tmpl.render(**context)
