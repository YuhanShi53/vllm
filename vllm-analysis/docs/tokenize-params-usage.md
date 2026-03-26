# TokenizeParams 自定义参数使用指南

本文档介绍如何在 vLLM serve 中通过 API 请求自定义 tokenization 参数。

## 概述

`TokenizeParams` 是控制 prompt 如何被 tokenization 的配置对象。在 vLLM serve 中，这些参数可以通过 API 请求体动态调整，每个请求可以独立设置不同的参数值。

## 参数定义位置

- **核心定义**: [vllm/renderers/params.py:77](../vllm/renderers/params.py#L77)
- **Completion API**: [vllm/entrypoints/openai/completion/protocol.py](../vllm/entrypoints/openai/completion/protocol.py)
- **Chat Completion API**: [vllm/entrypoints/openai/chat_completion/protocol.py](../vllm/entrypoints/openai/chat_completion/protocol.py)
- **Tokenize API**: [vllm/entrypoints/serve/tokenize/protocol.py](../vllm/entrypoints/serve/tokenize/protocol.py)

## 可配置参数

### 1. truncate_prompt_tokens

控制输入过长时的行为策略。

**取值含义:**
- `None`: 禁用截断（默认），输入过长时抛出 `VLLMValidationError`
- `-1`: 自动截断到 `max_input_tokens`（模型允许的最大输入长度）
- `k` (正整数): 保留最后 k 个 token（左截断）

**适用接口:** Completion API, Chat Completion API

**示例:**
```bash
# 不截断（默认）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 100
  }'

# 自动截断
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 100,
    "truncate_prompt_tokens": -1
  }'

# 精确控制（保留最后 2048 个 token）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 100,
    "truncate_prompt_tokens": 2048
  }'
```

**验证规则:**
- 不能超过 `max_model_len`（[serving.py:534-546](../vllm/entrypoints/openai/engine/serving.py#L534-L546)）
- 不能超过 `max_total_tokens - max_output_tokens`（[params.py:156-163](../vllm/renderers/params.py#L156-L163)）

### 2. add_special_tokens

控制是否添加特殊 tokens（如 BOS、EOS 等）。

**接口差异:**

| API | 默认值 | 说明 |
|-----|--------|------|
| Completion | `True` | 直接处理原始文本，通常需要添加 BOS/EOS |
| Chat Completion | `False` | Chat template 通常已包含特殊 token，避免重复 |
| Tokenize | `True` | 与 Completion 类似，处理原始文本 |

**Completion API 示例:**
```bash
# 默认添加特殊 token
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "Hello, world!",
    "max_tokens": 100
  }'

# 明确不添加特殊 token
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "add_special_tokens": false
  }'
```

**Chat Completion API 示例:**
```bash
# 默认不添加（chat template 会处理）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# 强制添加（在 chat template 之上，可能导致重复）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "add_special_tokens": true
  }'
```

**Tokenize API 示例:**
```bash
curl http://localhost:8000/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "add_special_tokens": true
  }'
```

## Python 客户端示例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Completion API - 控制截断和特殊 token
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="很长的输入...",
    max_tokens=100,
    truncate_prompt_tokens=-1,  # 自动截断
    add_special_tokens=True,
)

# Chat Completion API - 不同请求不同策略
response1 = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B",
    messages=[{"role": "user", "content": "短文本"}],
    truncate_prompt_tokens=None,  # 不截断
)

response2 = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B",
    messages=[{"role": "user", "content": "长文本..."}],
    truncate_prompt_tokens=2048,  # 精确控制
    add_special_tokens=False,  # 使用默认值
)

# Tokenize API
response = client.tokenize.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Hello, world!",
    add_special_tokens=True,
)
```

## 长度检查机制

vLLM 在 tokenization 之前会进行字符级预检查，以节省资源（避免对明显过长的输入进行昂贵的分词操作）。

**检查逻辑** ([params.py:243-267](../vllm/renderers/params.py#L243-L267)):

1. **如果没有设置 `max_input_tokens`**: 跳过检查
2. **如果 `truncate_prompt_tokens` 为 None 且 tokenizer 可用**:
   - 计算: `max_input_chars = max_input_tokens * tokenizer.max_chars_per_token`
   - 如果 `len(text) > max_input_chars`: 抛出 `VLLMValidationError`

**计算公式:**
```
max_input_tokens = max_total_tokens - max_output_tokens
max_input_chars = max_input_tokens * tokenizer.max_chars_per_token
```

**示例:**
```
假设:
- max_total_tokens = 4096
- max_output_tokens = 512
- max_input_tokens = 4096 - 512 = 3584
- tokenizer.max_chars_per_token = 5

如果输入文本长度 = 20000 字符:
- max_input_chars = 3584 * 5 = 17920
- 20000 > 17920 → 抛出异常
```

**注意:** 如果设置了 `truncate_prompt_tokens`，则跳过预检查（允许截断而非报错）。

## 参数传递流程

1. **API 请求** → 用户在请求体中设置参数
2. **Protocol 验证** → [protocol.py](../vllm/entrypoints/openai/chat_completion/protocol.py) 定义和验证参数
3. **构建 TokenizeParams** → 调用 `build_tok_params()` 方法
4. **Tokenization** → [base.py:241-252](../vllm/renderers/base.py#L241-L252) 使用参数进行分词

## 关键特性

✅ **完全动态**: 每个请求可以独立设置不同的参数值
✅ **实时生效**: 无需重启服务
✅ **灵活控制**: 支持严格模式、自动截断或精确控制
✅ **向后兼容**: 所有参数都有合理的默认值

## 相关文档

- [vLLM Input Preprocessing](../vllm/inputs/preprocess.py)
- [OpenAI Protocol](../vllm/entrypoints/openai/)
- [Renderer Base](../vllm/renderers/base.py)
- [Sampling Parameters](../vllm/sampling_params.py)

## 参考资料

- TokenizeParams 定义: [vllm/renderers/params.py:77](../vllm/renderers/params.py#L77)
- Length check: [vllm/renderers/params.py:243](../vllm/renderers/params.py#L243)
- Tokenization flow: [vllm/renderers/base.py:241](../vllm/renderers/base.py#L241)
