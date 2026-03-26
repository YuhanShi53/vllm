# mm_processor_kwargs 用法与内部处理

`mm_processor_kwargs` 是 vLLM 中用于向多模态处理器（HuggingFace `AutoProcessor`）传递额外参数的机制。它允许用户自定义图像、视频等多模态数据的预处理行为。

## 传参方法

### Chat Completion API 用法

**基本用法：**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="microsoft/Phi-3-vision-128k-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ],
    # 传递 processor 参数
    mm_processor_kwargs={"num_crops": 4}
)

print(response.choices[0].message.content)
```

**批量图片处理：**
```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "分析这些图片的差异"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg"}
                }
            ]
        }
    ],
    mm_processor_kwargs={
        "max_pixels": 2360,  # 控制最大像素数
        "min_pixels": 313    # 控制最小像素数
    }
)
```

**视频处理：**
```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这个视频在讲什么？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/video.mp4"}
                }
            ]
        }
    ],
    mm_processor_kwargs={
        "max_pixels": 2360,
        "fps": 2,  # 每秒提取帧数
        "max_frames": 32  # 最大帧数
    }
)
```

### LLM 类中的用法

**初始化时设置默认参数：**
```python
from vllm import LLM

# 初始化时设置全局默认参数
llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    mm_processor_kwargs={"num_crops": 4},  # 全局默认
    trust_remote_code=True
)

outputs = llm.generate({
    "prompt": "What's in this image?",
    "multi_modal_data": {
        "image": "https://example.com/image.jpg"
    }
    # 使用初始化时的 mm_processor_kwargs
})
```

**在 generate 调用中覆盖：**
```python
llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    trust_remote_code=True
)

outputs = llm.generate({
    "prompt": "详细分析这张图片",
    "multi_modal_data": {
        "image": "path/to/local/image.jpg"
    },
    # 覆盖默认参数
    "mm_processor_kwargs": {
        "num_crops": 16,  # 更高的裁剪数量，更精细
        "do_resize": True,
        "size": 448
    }
})
```

**批量生成：**
```python
llm = LLM(
    model="Qwen/Qwen2-VL-7B-Instruct",
    trust_remote_code=True
)

prompts = [
    {
        "prompt": "描述图片1",
        "multi_modal_data": {"image": "image1.jpg"},
        "mm_processor_kwargs": {"max_pixels": 2360}
    },
    {
        "prompt": "描述图片2",
        "multi_modal_data": {"image": "image2.jpg"},
        "mm_processor_kwargs": {"max_pixels": 5120}  # 不同图片用不同参数
    }
]

outputs = llm.generate(prompts)
```

**离线批处理：**
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="microsoft/Phi-3-vision-128k-instruct",
    mm_processor_kwargs={"num_crops": 4},
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100
)

# 批量处理
inputs = [
    {
        "prompt": f"What's in image {i}?",
        "multi_modal_data": {"image": f"images/image_{i}.jpg"}
    }
    for i in range(10)
]

outputs = llm.generate(inputs, sampling_params=sampling_params)
```

## 内部处理流程和参数去向

### 1. API 请求接收

**Chat Completion 协议层：**
```python
# vllm/entrypoints/openai/chat_completion/protocol.py:270-273
class ChatCompletionRequest(OpenAIBaseModel):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
```

**Responses 协议层：**
```python
# vllm/entrypoints/openai/responses/protocol.py:200-203
class ResponsesRequest(OpenAIBaseModel):
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
```

### 2. Serving 层处理

**OpenAIServing 提取参数：**
```python
# vllm/entrypoints/openai/engine/serving.py:990-996
prompt_extras={
    k: v
    for k in ("mm_processor_kwargs", "cache_salt")
    if (v := getattr(request, k, None)) is not None
}
```

参数被放入 `prompt_extras` 字典，传递给 tokenizer。

### 3. Prompt 预处理层

**从 Prompt 中提取参数：**
```python
# vllm/inputs/preprocess.py:233
# 在 _process_tokens 方法中
parsed_content.get("mm_processor_kwargs") or {}

# vllm/inputs/preprocess.py:259
# 在 _process_text 方法中
parsed_content.get("mm_processor_kwargs") or {}
```

### 4. 多模态处理器应用

**传递给 HF Processor：**
```python
# vllm/inputs/preprocess.py:158-166
def _process_multimodal(
    self,
    prompt: str | list[int],
    mm_data: MultiModalDataDict,
    mm_processor_kwargs: dict[str, Any] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    *,
    mm_uuids: MultiModalUUIDDict | None = None,
) -> MultiModalInputs:
    mm_processor = self.renderer.get_mm_processor()

    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}

    mm_items = mm_processor.info.parse_mm_data(mm_data)

    return mm_processor.apply(
        prompt,
        mm_items,
        hf_processor_mm_kwargs=mm_processor_kwargs,  # 关键传递点
        tokenization_kwargs=tokenization_kwargs,
        mm_uuids=mm_uuids,
    )
```

### 5. 多模态处理核心

**应用 HF Processor 并缓存：**
```python
# vllm/multimodal/processing/processor.py:1514-1597
def _cached_apply_hf_processor(
    self,
    prompt: str | list[int],
    mm_data_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],  # 接收参数
    tokenization_kwargs: Mapping[str, object],
    *,
    mm_uuids: MultiModalUUIDDict | None = None,
) -> tuple[list[int], MultiModalProcessingInfo, bool]:
    """Apply the HF processor on the full prompt text,
    caching the results and reusing cached results."""

    # 计算哈希（包含 mm_processor_kwargs）
    mm_hashes = self._hash_mm_items(
        mm_data_items,
        hf_processor_mm_kwargs,  # 影响缓存键
        tokenization_kwargs,
        mm_uuids=mm_uuids,
    )

    # 检查缓存
    mm_is_cached, mm_missing_data_items = self._get_cache_missing_items(
        cache=cache,
        mm_data_items=mm_data_items,
        mm_hashes=mm_hashes,
    )

    # 应用 processor
    return self._apply_hf_processor(
        prompt=prompt,
        mm_data_items=mm_data_items,
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,  # 传递给 HF processor
        tokenization_kwargs=tokenization_kwargs,
        mm_uuids=mm_uuids,
    )
```

**最终传递给 HuggingFace Processor：**
```python
# vllm/multimodal/processing/processor.py:1400-1420
def _apply_hf_processor_main(
    self,
    prompt: str | list[int],
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, object],
    tokenization_kwargs: Mapping[str, object],
    enable_hf_prompt_update: bool = True,
):
    # ... 省略代码 ...

    # 调用 HuggingFace processor
    processed_outputs = self.info.hf_processor(
        text=text_inputs,
        images=mm_kwargs.get("images"),
        videos=mm_kwargs.get("videos"),
        audio=mm_kwargs.get("audio"),
        **hf_processor_mm_kwargs,  # 最终传递到这里
    )

    return processed_outputs
```

### 参数传递流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     API 请求层                               │
├─────────────────────────────────────────────────────────────┤
│ ChatCompletionRequest.mm_processor_kwargs                   │
│ {"num_crops": 4}                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Serving 层                                │
├─────────────────────────────────────────────────────────────┤
│ OpenAIServing._create_chat_completion                       │
│   → prompt_extras = {                                       │
│       "mm_processor_kwargs": {"num_crops": 4}               │
│     }                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Renderer 层                                 │
├─────────────────────────────────────────────────────────────┤
│ Renderer.tokenize_prompt                                    │
│   → prompt_extras 传入                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                Input Preprocessor 层                         │
├─────────────────────────────────────────────────────────────┤
│ InputPreprocessor._process_multimodal                       │
│   → parsed_content.get("mm_processor_kwargs")               │
│   → mm_processor.apply(                                     │
│       hf_processor_mm_kwargs={"num_crops": 4}               │
│     )                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            MultiModal Processor 层                          │
├─────────────────────────────────────────────────────────────┤
│ MultiModalProcessor._cached_apply_hf_processor             │
│   → 计算缓存哈希（包含 mm_processor_kwargs）                 │
│   → 检查缓存                                                 │
│   → 调用 HuggingFace processor                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          HuggingFace Processor 执行                         │
├─────────────────────────────────────────────────────────────┤
│ processor(                                                 │
│     text=prompt,                                            │
│     images=images,                                          │
│     **hf_processor_mm_kwargs  # {"num_crops": 4}           │
│ )                                                           │
└─────────────────────────────────────────────────────────────┘
```

### 缓存机制

`mm_processor_kwargs` **影响缓存键**。不同的参数组合会产生不同的缓存条目：

```python
# vllm/multimodal/processing/processor.py:1540-1545
mm_hashes = self._hash_mm_items(
    mm_data_items,
    hf_processor_mm_kwargs,  # 参与哈希计算
    tokenization_kwargs,
    mm_uuids=mm_uuids,
)
```

这意味着：
- `{"num_crops": 4}` 和 `{"num_crops": 16}` 会产生不同的缓存
- 相同的参数会重用缓存，提高性能

## 常见模型的 mm_processor_kwargs 参数

### Phi-3-Vision
```python
mm_processor_kwargs = {
    "num_crops": 4,     # 图像裁剪数量 (默认 4，最大 16)
}
```

### Qwen2-VL / Qwen2.5-VL
```python
mm_processor_kwargs = {
    "max_pixels": 2360,   # 最大像素数
    "min_pixels": 313,    # 最小像素数
    "fps": 2,             # 视频帧率
    "max_frames": 32,     # 最大帧数
}
```

### LLaVA 模型系列
```python
mm_processor_kwargs = {
    "do_resize": True,        # 是否调整大小
    "size": 336,              # 目标尺寸
    "do_center_crop": True,   # 是否中心裁剪
    "crop_size": 336,         # 裁剪尺寸
}
```

### Pixtral
```python
mm_processor_kwargs = {
    "do_resize": True,
    "size": {
        "height": 1024,
        "width": 1024
    }
}
```

## 注意事项

1. **参数模型相关**：不同模型支持的参数不同，需参考对应的 processor 文档
2. **性能影响**：不同的参数值会创建不同的缓存条目
3. **覆盖优先级**：请求级参数 > 初始化级参数
4. **类型检查**：必须是 `dict[str, Any]` 类型
5. **安全性**：只允许处理允许的媒体源（本地路径或域名）

## 相关文件

- [Chat Completion 协议](vllm/entrypoints/openai/chat_completion/protocol.py:270-273)
- [Responses 协议](vllm/entrypoints/openai/responses/protocol.py:200-203)
- [Serving 层](vllm/entrypoints/openai/engine/serving.py:990-996)
- [Input 预处理](vllm/inputs/preprocess.py:158-166)
- [多模态处理器](vllm/multimodal/processing/processor.py:1514-1597)
- [LLM 类](vllm/entrypoints/llm.py:178-182)
