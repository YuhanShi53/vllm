# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation

**Python-only development (modify Python code only):**
```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

**Full build from source (modify C++/CUDA kernels):**
```bash
uv pip install -e .
```

**Install with existing PyTorch:**
```bash
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

### Testing

```bash
# Install test dependencies
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Run all tests
pytest tests/

# Run single test file with verbose output
pytest -s -v tests/test_logger.py

# Run tests with specific markers
pytest -m "not slow_test" tests/
pytest -m "core_model" tests/
pytest -m "distributed" tests/
```

### Linting and Formatting

```bash
# Install pre-commit hooks (runs automatically on commit)
uv pip install pre-commit
pre-commit install

# Run pre-commit manually
pre-commit run              # on staged files
pre-commit run -a           # on all files

# Run specific hooks manually
pre-commit run --hook-stage manual mypy-3.12

# Individual tools
ruff check .                # lint Python
ruff format .               # format Python
clang-format --style=file   # format C++/CUDA
```

### Documentation

```bash
uv pip install -r requirements/docs.txt
mkdocs serve                           # with API reference (~10 min)
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # without API reference (~15 sec)
```

## High-Level Architecture

vLLM is a high-throughput LLM inference and serving engine built around **PagedAttention**, a novel memory management technique that treats KV cache as virtual memory pages.

### Core Components

**Entrypoints** (`vllm/entrypoints/`):
- `LLM` class (`llm.py`) - Primary Python interface for offline inference
- `AsyncLLMEngine` wrapper (`openai/api_server.py`) - OpenAI-compatible API server
- CLI (`cli/main.py`) - `vllm serve` command

**Engine** (`vllm/engine/`):
- `LLMEngine` (`llm_engine.py`) - Core synchronous inference engine
- `AsyncLLMEngine` (`async_llm_engine.py`) - Asynchronous wrapper for online serving
- Handles tokenization, scheduling, model execution, output processing

**Model Executor** (`vllm/model_executor/`):
- `ModelRunner` - Loads and runs models, prepares input tensors, captures CUDA graphs
- Model implementations for 50+ architectures (transformer LLMs, MoE, embedding models, multimodal)
- Custom CUDA kernels in `csrc/` (attention, quantization, MoE, Mamba)

**Scheduler** (`vllm/scheduler/`):
- Implements continuous batching for dynamic request scheduling
- Manages PagedAttention memory allocation
- Prefix caching optimization

**Configuration** (`vllm/config/`):
- `VllmConfig` - Central configuration object passed to all components
- Model, cache, parallel, scheduler, and hardware-specific configs

### Key Design Patterns

1. **Uniform Configuration**: All classes accept a `VllmConfig` object containing all necessary information. This enables extensibility - new features only require adding config options, not changing constructor signatures.

2. **Uniform Model Constructors**: All models use keyword-only `__init__(*, vllm_config: VllmConfig, prefix: str = "")` for consistent instantiation across 50+ architectures.

3. **Sharding at Initialization**: Tensor parallelism and quantization modify weights during model initialization (not after) to reduce memory overhead for large models.

4. **Worker Process Model**: One process per accelerator device (GPU), identified by `rank` (global) and `local_rank` (device assignment).

### Directory Structure

```
vllm/                    # Python source
  entrypoints/           # API servers and CLI
  engine/                # Core inference engine
  model_executor/        # Model execution and layers
  attention/             # PagedAttention implementation
  distributed/           # Multi-GPU/multi-node support
  config/                # Configuration classes
  worker/                # Worker process logic
  transformers_utils/    # HuggingFace integration

csrc/                    # CUDA/C++ kernels
  attention/             # Attention kernels
  quantization/          # Quantization kernels
  moe/                   # Mixture-of-Experts kernels
  cache/                 # KV cache operations

tests/                   # Pytest tests
  basic_correctness/     # Core functionality tests
  distributed/           # Multi-GPU tests
  models/                # Model-specific tests
  kernels/               # Kernel unit tests
```

### Performance Optimizations

- **PagedAttention** - Memory-efficient KV cache management
- **Continuous batching** - Dynamic request scheduling
- **CUDA graphs** - Kernel fusion and reduced launch overhead
- **Speculative decoding** - Draft model acceleration
- **Quantization** - INT4, INT8, FP8, GPTQ, AWQ, AutoRound
- **Prefix caching** - Reuse cached KV cache across requests

### Important Notes

- Python 3.10-3.13 supported; CI uses Python 3.12
- Pre-built wheels available for every commit on `main` (wheels.vllm.ai)
- For C++/CUDA kernel development, use `ccache` or the Incremental Compilation Workflow (`docs/contributing/incremental_build.md`)
- vLLM requires Linux; use WSL for Windows development
- `vllm/third_party/` is excluded from linting
- Generated gRPC protobuf files (`*_pb2.py`) are excluded from linting

### Code Quality Standards

- Google Python Style Guide and Google C++ Style Guide
- All Python code must pass ruff, mypy (partial coverage)
- All C++/CUDA code must pass clang-format
- New models require registration in `vllm/model_executor/models/registry.py`
- Pre-commit hooks require DCO sign-off (`git commit -s`)
- PR titles use prefixes: `[Bugfix]`, `[Model]`, `[Kernel]`, `[Core]`, `[Frontend]`, `[CI/Build]`, `[Doc]`, `[Hardware][Vendor]`
