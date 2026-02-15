# vLLM 学习指导

## 架构

├── 架构层次分析
│   ├── 2.1 配置层
│   │   ├── 核心类: VllmConfig (统一配置对象)
│   │   ├── 子配置模块:
│   │   │   ├── ModelConfig - 模型架构、类型、量化
│   │   │   ├── CacheConfig - KV缓存参数
│   │   │   ├── ParallelConfig - 并行设置
│   │   │   ├── SchedulerConfig - 调度策略
│   │   │   ├── LoadConfig - 模型加载
│   │   │   ├── LoRAConfig - LoRA适配器
│   │   │   └── MultiModalConfig - 多模态配置
│   │   └── 设计原则: 配置贯穿所有层
│   │
│   ├── 2.2 入口层
│   │   ├── LLM 类 - Python API (离线推理)
│   │   ├── OpenAI API 服务器 (在线服务)
│   │   └── CLI 工具 (generate, complete)
│   │
│   ├── 2.3 引擎层
│   │   ├── LLMEngine (同步引擎)
│   │   │   ├── 调度决策
│   │   │   ├── 模型执行
│   │   │   └── I/O 处理
│   │   │
│   │   ├── AsyncLLMEngine (异步引擎)
│   │   │   └── 流式输出支持
│   │   │
│   │   └── V1 引擎
│   │       └── input_processor.py (你打开的文件)
│   │           ├── 输入预处理
│   │           ├── 参数验证
│   │           ├── 多模态数据处理
│   │           └── 结构化输出支持
│   │
│   ├── 2.4 工作器层
│   │   ├── 每个 GPU 一个 worker
│   │   ├── 标识: rank (全局), local_rank (设备)
│   │   └── 执行分配的推理任务
│   │
│   ├── 2.5 模型运行器
│   │   ├── 每个 worker 一个
│   │   ├── 模型加载
│   │   ├── 输入准备
│   │   ├── CUDA Graph 捕获
│   │   └── 实际模型执行
│   │
│   └── 2.6 模型层
│       ├── torch.nn.Module 实现
│       ├── 支持 50+ 模型类型
│       ├── 统一构造函数: __init__(self, *, vllm_config: VllmConfig)
│       └── 常见模型: Llama, GPT, Mistral, Mixtral

## 关键代码路径

├── 关键代码路径
│   ├── 3.1 目录结构
│   │   ├── vllm/config/ - 配置定义
│   │   ├── vllm/engine/ - 核心引擎
│   │   ├── vllm/v1/engine/ - V1 新引擎
│   │   ├── vllm/worker/ - 工作器实现
│   │   ├── vllm/model_executor/ - 模型执行器
│   │   │   ├── models/ - 模型定义
│   │   │   ├── layers/ - 神经网络层
│   │   │   └── model_loader.py - 模型加载
│   │   ├── vllm/distributed/ - 分布式执行
│   │   ├── vllm/multimodal/ - 多模态支持
│   │   ├── vllm/attention/ - 注意力实现
│   │   ├── vllm/lora/ - LoRA 支持
│   │   └── csrc/ - CUDA/C++ 内核
│   │
│   ├── 3.2 关键文件
│   │   ├── vllm/config/vllm.py - VllmConfig (配置核心)
│   │   ├── vllm/engine/llm_engine.py - 主引擎
│   │   ├── vllm/v1/engine/input_processor.py - 输入处理
│   │   ├── vllm/model_executor/models/registry.py - 模型注册
│   │   ├── csrc/attention/attention_kernels.cu - 注意力内核
│   │   └── vllm/worker/worker.py - 工作器基类
│   │
│   └── 3.3 数据流
│       ├── 请求输入 → 引擎
│       ├── 引擎 → 调度器 (决策批次)
│       ├── 调度器 → Worker (分配任务)
│       ├── Worker → 模型运行器 (准备输入)
│       ├── 模型运行器 → 模型 (执行推理)
│       └── 模型 → 输出 (返回结果)

## 底层原理

├── 底层原理
│   ├── 4.1 调度机制
│   │   ├── 决策: 哪些请求处理
│   │   ├── KV Cache 分配
│   │   ├── 优先级处理
│   │   └── 抢占机制
│   │
│   ├── 4.2 多模态处理
│   │   ├── 输入: 视觉、音频、视频
│   │   ├── 多模态注册表 (registry)
│   │   ├── 特征提取
│   │   └── 与文本融合
│   │
│   ├── 4.3 量化技术
│   │   ├── GPTQ, AWQ
│   │   ├── FP8, INT8, INT4
│   │   ├── 量化内核 (csrc/quantization/)
│   │   └── 性能与精度平衡
│   │
│   ├── 4.4 LoRA 与适配器
│   │   ├── 高效模型微调
│   │   ├── 文件系统解析器
│   │   └── 前缀缓存
│   │
│   └── 4.5 投机解码
│       ├── Draft Model
│       ├── 验证机制
│       └── 加速推理

## CUDA

├── CUDA
│   ├── 5.1 自定义内核位置
│   │   ├── csrc/attention/ - PagedAttention
│   │   ├── csrc/core/ - 核心工具
│   │   ├── csrc/quantization/ - 量化内核
│   │   ├── csrc/moe/ - 混合专家
│   │   └── csrc/cpu/ - CPU 优化
│   │
│   ├── 5.2 开发流程
│   │   ├── 编辑内核代码
│   │   ├── 生成 CMake 预设
│   │   ├── 增量构建
│   │   └── 测试验证
│   │
│   └── 5.3 性能优化
│       ├── AMX, AVX-512 (CPU)
│       ├── Tensor Core 利用
│       └── 内存访问优化

## 学习路径

└── 学习路径建议
    ├── 8.1 入门阶段
    │   ├── 阅读 CLAUDE.md
    │   ├── 理解基本概念 (PagedAttention, Continuous Batching)
    │   ├── 运行示例代码
    │   └── 熟悉配置系统
    │
    ├── 8.2 进阶阶段
    │   ├── 分析引擎流程 (engine/llm_engine.py)
    │   ├── 研究调度器实现
    │   ├── 理解模型加载 (model_loader.py)
    │   └── 学习分布式通信 (distributed/)
    │
    ├── 8.3 深入阶段
    │   ├── 阅读 CUDA 内核代码
    │   ├── 研究量化实现
    │   ├── 理解 MoE 路由
    │   └── 分析多模态处理
    │
    └── 8.4 实践阶段
        ├── 尝试添加新模型
        ├── 修改现有功能
        ├── 优化内核性能
        └── 贡献 PR
