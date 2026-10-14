from llmhub.core.template import ChatTemplateEngine

def test_chatml_fallback():
    engine = ChatTemplateEngine(template_str=None, architecture="qwen")
    messages = [
        {"role": "system", "content": "You are a bot."},
        {"role": "user", "content": "Hello!"}
    ]
    prompt = engine.render(messages, add_generation_prompt=True)
    assert "<|im_start|>system\nYou are a bot.<|im_end|>" in prompt
    assert "<|im_start|>user\nHello!<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

def test_llama3_fallback():
    engine = ChatTemplateEngine(template_str=None, architecture="llama")
    messages = [
        {"role": "user", "content": "Hi Llama"}
    ]
    prompt = engine.render(messages, add_generation_prompt=True)
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt

def test_custom_template():
    custom = "[INST] {% for msg in messages %}{{ msg['content'] }} {% endfor %}[/INST]"
    engine = ChatTemplateEngine(template_str=custom)
    messages = [{"role": "user", "content": "Explain CPU caching."}]
    rendered = engine.render(messages)
    assert rendered.strip() == "[INST] Explain CPU caching. [/INST]"
