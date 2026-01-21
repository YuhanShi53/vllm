# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## vLLM Architecture Overview

vLLM is a fast and easy-to-use library for LLM inference and serving, originally developed at UC Berkeley. The codebase has undergone a major architectural upgrade with the **V1 engine** (announced January 2025), which provides ~1.7x speedup over the legacy design.

### V1 vs Legacy Architecture

**Important:** The legacy `vllm/engine/` directory now simply redirects to V1 implementations. The primary development happens in the V1 architecture.

- **V1 Engine:** `vllm/v1/engine/llm_engine.py` - Main entry point
- **Legacy redirect:** `vllm/engine/llm_engine.py` - Just re-exports V1

### Key V1 Architecture Components

```
vllm/v1/
├── engine/
│   ├── llm_engine.py    # Main LLMEngine (front-facing API)
│   ├── core.py           # EngineCore - inner computation loop
│   ├── input_processor.py  # Process incoming prompts
│   └── output_processor.py # Convert EngineCoreOutputs -> RequestOutput
├── worker/               # Worker implementations (GPU/TPU/CPU)
├── core/
│   └── sched/            # Scheduler implementations
├── executor/             # Execution backends (multi-GPU, distributed)
├── attention/            # Attention mechanisms
├── sample/               # Sampling logic
├── metrics/              # Metrics and stats collection
└── kv_cache_interface/   # KV cache management
```

### Core Concepts

1. **PagedAttention:** Efficient KV cache management using paged memory (key innovation)
2. **Continuous Batching:** Processes multiple requests simultaneously with dynamic batching
3. **Decoupled Engine Design (V1):** `EngineCore` handles computation separately from `LLMEngine`
4. **Prefix Caching:** Zero-overhead caching of common prompt prefixes

### Parallelism Types

vLLM supports four types of parallelism (configured in `vllm/config/parallel.py`):

- **Tensor Parallelism** (`tensor_parallel_size`): Shards model weights across GPUs
- **Pipeline Parallelism** (`pipeline_parallel_size`): Splits model layers across devices
- **Data Parallelism** (`data_parallel_size`): Replicates model across devices
- **Expert Parallelism** (`enable_expert_parallel`): For mixture-of-experts (MoE) models

## Development Commands

### Initial Setup

```bash
# Create virtual environment (Python 3.12 recommended - matches CI)
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM (Python-only development - uses precompiled kernels)
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto

# Install vLLM (full development - compiles CUDA/C++ kernels)
uv pip install -e .
```

### Incremental Build for CUDA/C++ Development

When working on kernels in `csrc/`, use the incremental build workflow:

```bash
# 1. Generate CMake presets (auto-detects CUDA, Python paths)
python tools/generate_cmake_presets.py

# 2. Configure CMake build
cmake --preset release

# 3. Build and install (updates your editable install)
cmake --build --preset release --target install

# 4. Make changes and repeat step 3
cmake --build --preset release --target install
```

The incremental build uses `ccache` for fast rebuilds and installs kernels directly into your source tree, updating the editable Python installation.

### Testing

```bash
# Install test dependencies
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Run specific test categories
pytest tests/basic_correctness/    # Core functionality
pytest tests/distributed/          # Distributed inference
pytest tests/v1/                   # V1 architecture tests
pytest tests/models/               # Model-specific tests

# Run with markers
pytest -m slow_test                # Run slow tests
pytest -m distributed              # Distributed tests only
pytest -m "not skip_v1"            # Exclude tests that skip V1
pytest -m "not slow_test"          # Exclude slow tests

# Run specific test file
pytest -s -v tests/test_logger.py

# Run optional tests
pytest --optional
```

### Linting and Formatting

```bash
# Install pre-commit hooks
uv pip install pre-commit
pre-commit install

# Run manually on staged files
pre-commit run

# Run on all files
pre-commit run -a

# Run specific hooks (manual-stage hooks)
pre-commit run --hook-stage manual markdownlint
pre-commit run --hook-stage manual mypy-3.10
```

vLLM uses:
- **ruff** for linting (configured in `pyproject.toml`)
- **mypy** for type checking (run in manual stage)
- **typos** for spell checking

### Documentation

```bash
# Install docs dependencies
uv pip install -r requirements/docs.txt

# Serve docs locally (with API ref - takes ~10 min)
mkdocs serve

# Serve docs locally (without API ref - ~15 sec)
API_AUTONAV_EXCLUDE=vllm mkdocs serve
```

Visit http://127.0.0.1:8000/ for live preview.

### Running vLLM

```bash
# Start OpenAI-compatible API server
vllm serve meta-llama/Llama-3.1-8B

# Or via Python module
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B
```

## Code Organization

### Model Executor

```
vllm/model_executor/
├── models/              # Model-specific implementations
├── layers/              # Common layer implementations
├── parallel_utils/      # Parallel communication utilities
└── model_loader.py      # Model loading logic
```

### Configuration

- `vllm/config/` - All configuration classes
- `vllm/config/parallel.py` - ParallelConfig (tensor/pipeline/data/expert parallelism)
- `VllmConfig` - Top-level config that aggregates all sub-configs

### Entry Points

- `vllm/entrypoints/llm.py` - `LLM` class (Python API)
- `vllm/entrypoints/openai/api_server.py` - OpenAI-compatible server
- `vllm/entrypoints/cli/main.py` - `vllm` CLI command

### Multimodal Support

- `vllm/multimodal/` - Multimodal infrastructure
- `MULTIMODAL_REGISTRY` - Registry for multimodal models (e.g., LLaVA)

### LoRA Support

- `vllm/lora/` - LoRA (Low-Rank Adaptation) adapter support
- `vllm/plugins/lora_resolvers/` - LoRA filesystem resolvers

## PR Submission Guidelines

### PR Title Prefixes

When submitting PRs, use these prefixes:
- `[Bugfix]` - Bug fixes
- `[CI/Build]` - Build or CI improvements
- `[Doc]` - Documentation fixes
- `[Model]` - New model support (model name in title)
- `[Frontend]` - Changes to OpenAI API, LLM class, etc.
- `[Kernel]` - CUDA/C++ kernel changes
- `[Core]` - Core engine logic (LLMEngine, Scheduler, etc.)
- `[Hardware][Vendor]` - Hardware-specific changes (e.g., `[Hardware][AMD]`)
- `[Misc]` - Other changes (use sparingly)

### DCO (Developer Certificate of Origin)

Commits must include `Signed-off-by:` header. Use `-s` flag:
```bash
git commit -s -m "Your commit message"
```

### Custom Kernels

When adding or modifying kernels:
1. Follow PyTorch custom op guidelines: [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)
2. Custom ops returning Tensors require meta-functions (implement in Python)
3. Use `torch.library.opcheck()` to test function registration
4. See `tests/kernels` for examples

### Major Changes

For architectural changes >500 LOC (excluding kernel/data/config/test):
- File a GitHub issue (RFC) discussing the technical design first
- Otherwise, PR may be tagged with `rfc-required`

## Test Markers

Tests use pytest markers (defined in `pyproject.toml`):
- `slow_test` - Slow-running tests
- `skip_v1` - Tests that should not run with V1
- `distributed` - Distributed GPU tests
- `cpu_test` - CPU-only tests
- `optional` - Optional tests (require `--optional` flag)
- `core_model` - Models enabled in each PR (not just nightly)
- `hybrid_model` - Models with mamba layers
- `cpu_model` - Models enabled in CPU tests

## Key Dependencies

- **PyTorch 2.9.0** (pinned version)
- **CMake >= 3.26.1** + **Ninja** for building C++/CUDA
- **setuptools** + **setuptools-scm** for Python packaging
- **ccache** (recommended) for fast incremental builds

## Important Notes

1. **V1 is the default:** The `vllm/engine/` imports now redirect to V1 implementations. New development should target V1.
2. **Python 3.12 recommended:** CI runs with Python 3.12, so developing with 3.12 minimizes environment mismatches.
3. **CPU tests:** Not all tests pass on CPU - rely on CI for full test coverage if GPU unavailable.
4. **PagedAttention:** The core innovation - KV cache management uses paged memory (similar to OS virtual memory).
5. **Multi-LoRA:** vLLM supports serving multiple LoRA adapters concurrently.

## Resources

- Documentation: https://docs.vllm.ai
- Contributing guide: `docs/contributing/README.md`
- Incremental build: `docs/contributing/incremental_build.md`
- GitHub Issues: https://github.com/vllm-project/vllm/issues
- Slack: https://slack.vllm.ai
- User Forum: https://discuss.vllm.ai
