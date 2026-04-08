# LLMRouterBench 数据收集指南

本文档记录了 LLMRouterBench 项目的数据收集过程，包括环境配置、问题修复、执行步骤和结果验证，便于复现和参考。

## 目录

- [概述](#概述)
- [前置要求](#前置要求)
- [环境配置](#环境配置)
- [问题修复](#问题修复)
- [数据收集流程](#数据收集流程)
- [结果验证](#结果验证)
- [常见问题](#常见问题)
- [性能优化](#性能优化)

---

## 概述

### 项目目标

为 Local vs Cloud LLM Router 收集推理数据，用于训练路由模型。路由器在以下两个模型之间进行选择：

- **Local 模型**: `qwen3-coder-next-80b-4bit-awq` (代码专用，成本低)
- **Cloud 模型**: `qwen3.5-35b-a3b-awq-4b` (通用 MoE，成本高)

### 数据集

收集 4 个数据集的推理结果：

| 数据集 | Split | 记录数 | 说明 |
|--------|-------|--------|------|
| AIME | hybrid | 60 | 数学竞赛题 |
| LiveMathBench | test | 121 | 数学推理 |
| GPQA | test | 198 | 研究生级别问答 |
| MMLU-Pro | test_3000 | 3000 | 多领域选择题 |

**注**: LiveCodeBench 需要单独从 HuggingFace 下载数据，本次暂不收集。

---

## 前置要求

### 系统环境

- **操作系统**: Windows 10/11 (或 Linux/macOS)
- **Python**: 3.8+
- **Docker**: 用于运行 MySQL 缓存服务
- **磁盘空间**: 至少 5GB（用于存储结果 JSON 文件）

### 依赖安装

```bash
# 创建 Python 虚拟环境
python -m venv venv_router

# 激活虚拟环境
# Windows:
venv_router\Scripts\activate
# Linux/macOS:
source venv_router/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### API 密钥

需要 Intel SuperRouter API 密钥。在 `.env` 文件中配置：

```bash
INTEL_SUPERROUTER_API_KEY=sk-your-api-key-here
```

---

## 环境配置

### 1. 启动 MySQL Docker 容器

数据收集工具使用 MySQL 作为缓存层，避免重复的 API 调用。

```bash
# 启动容器
docker start llmrouterbench-mysql

# 验证运行状态
docker ps | grep mysql
# 输出应包含: llmrouterbench-mysql ... Up ... 3306/tcp
```

如果容器不存在，需要先创建：

```bash
docker run -d \
  --name llmrouterbench-mysql \
  -e MYSQL_ROOT_PASSWORD=your_password \
  -e MYSQL_DATABASE=local_vs_cloud_cache \
  -p 3306:3306 \
  mysql:8.0
```

### 2. 配置环境变量

SuperRouter 使用自签名 SSL 证书，需要配置代理绕过：

```bash
# Windows CMD:
set NO_PROXY=superrouter.intel.com,localhost,127.0.0.1
set no_proxy=superrouter.intel.com,localhost,127.0.0.1

# Windows PowerShell:
$env:NO_PROXY="superrouter.intel.com,localhost,127.0.0.1"
$env:no_proxy="superrouter.intel.com,localhost,127.0.0.1"

# Linux/macOS:
export NO_PROXY="superrouter.intel.com,localhost,127.0.0.1"
export no_proxy="superrouter.intel.com,localhost,127.0.0.1"
```

**重要**: 每次新开终端都需要重新设置这些环境变量。

---

## 问题修复

在数据收集过程中遇到并解决了以下问题：

### 问题 1: SSL 证书验证失败

**错误信息**:
```
[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain
```

**原因**: SuperRouter API 使用自签名 SSL 证书，Python 的 `httpx` 库默认拒绝连接。

**解决方案**: 修改 `generators/generator.py` 第 68 行，扩展 SSL 绕过条件：

```python
# 修改前:
if "21020" in self.base_url:

# 修改后:
if "21020" in self.base_url or "superrouter" in self.base_url:
```

**提交记录**: `5ce6aed - fix(generators): add SSL verify=False for SuperRouter`

### 问题 2: Windows GBK 编码错误

**错误信息**:
```
'gbk' codec can't decode byte 0x9d in position 5052: illegal multibyte sequence
```

**原因**: GPQA 数据集包含 UTF-8 字符，但 Windows 默认使用 GBK 编码读取文件。

**解决方案**: 修改 `evaluation/base_evaluator.py` 第 30 行，在 `load_jsonl()` 方法中显式指定编码：

```python
# 修改前:
with open(file_path, "r") as file:

# 修改后:
with open(file_path, "r", encoding="utf-8") as file:
```

**提交记录**: `e9591ab - fix(gpqa): add UTF-8 encoding for Windows compatibility`

### 问题 3: API 超时和重试优化

**现象**: SuperRouter API 不稳定，频繁出现超时（尤其在 GPQA 收集期间）。

**解决方案**: 
1. 设置超时时间为 300 秒
2. 启用重试机制（最多 10 次）
3. 调整并发数为 4（平衡速度和稳定性）

**配置**: 在 YAML 配置文件中设置：

```yaml
models:
  - timeout: 300  # 5 分钟超时

run:
  concurrency: 4  # 4 个并发请求
```

**提交记录**: `5e1de26 - perf(config): reduce concurrency to 4 for SuperRouter stability`

---

## 数据收集流程

### 步骤 1: 准备配置文件

创建数据收集配置文件 `config/data_collector_<model_name>.yaml`：

```yaml
models:
  - name: qwen3-coder-next
    api_model_name: qwen3-coder-next-80b-4bit-awq
    base_url: https://superrouter.intel.com/v1
    api_key: INTEL_SUPERROUTER_API_KEY
    temperature: 0.2
    top_p: 1.0
    timeout: 300
    pricing:
      prompt_price_per_million: 0.01
      completion_price_per_million: 0.01

datasets:
  - dataset_id: aime
    splits: ["hybrid"]
  - dataset_id: livemathbench
    splits: ["test"]
  - dataset_id: gpqa
    splits: ["test"]
  - dataset_id: mmlupro
    splits: ["test_3000"]

run:
  output_dir: ./results
  overwrite: false
  concurrency: 4
  log_level: INFO

cache:
  enabled: true
  force_override_cache: false
  mysql:
    host: MYSQL_HOST
    port: MYSQL_PORT
    user: MYSQL_USER
    password: MYSQL_PASSWORD
    database: local_vs_cloud_cache
```

### 步骤 2: 设置环境变量

```bash
# 设置 NO_PROXY（必须）
export NO_PROXY="superrouter.intel.com,localhost,127.0.0.1"
export no_proxy="superrouter.intel.com,localhost,127.0.0.1"

# 切换到项目目录
cd D:/router/LLMRouterBench
```

### 步骤 3: 运行数据收集器

#### 方式 A: 前台运行（实时查看日志）

```bash
D:/router/venv_router/Scripts/python.exe -m data_collector.cli run config/data_collector_<model_name>.yaml -y
```

**参数说明**:
- `run`: 执行数据收集
- `-y`: 跳过交互式确认（自动回答 "yes"）

#### 方式 B: 后台运行（推荐用于长时间收集）

```bash
nohup D:/router/venv_router/Scripts/python.exe -m data_collector.cli run config/data_collector_<model_name>.yaml -y > collection.log 2>&1 &
```

**查看进度**:

```bash
# 实时查看日志
tail -f collection.log

# 检查进程状态
wmic process where "CommandLine like '%data_collector%'" get ProcessId,CommandLine
```

### 步骤 4: 监控收集进度

日志中会显示进度信息：

```
[INFO] Starting execution of 4 runs
[INFO] Processing run: aime/hybrid/qwen3-coder-next
[INFO] Loaded 60 records for aime/hybrid
Processing records: 100%|██████████| 60/60 [12:29<00:00, 12.49s/record]
[INFO] Saved result: results/bench/aime/hybrid/qwen3-coder-next/aime-hybrid-qwen3-coder-next-20260311_204808.json
[INFO] Completed run aime/hybrid/qwen3-coder-next: 0.867 performance
```

**关键指标**:
- `Processing records`: 当前进度（已处理/总数）
- `s/record`: 每条记录的平均处理时间
- `performance`: 模型准确率（0.0-1.0）

### 步骤 5: 等待完成

不同数据集的预计时间：

| 数据集 | 记录数 | 预计时间 | 说明 |
|--------|--------|----------|------|
| AIME | 60 | 12-15 分钟 | 数学题，输出较长 |
| LiveMathBench | 121 | 40-50 分钟 | 数学推理 |
| GPQA | 198 | 3-6 小时 | API 不稳定，重试多 |
| MMLU-Pro | 3000 | 4-5 小时 | 数据量大但稳定 |

**总计**: 约 10-12 小时（取决于 API 稳定性）

---

## 结果验证

### 检查结果文件

收集完成后，验证结果文件是否生成：

```bash
# 列出所有结果目录
ls -R results/bench/

# 预期输出结构:
# results/bench/
#   aime/hybrid/qwen3-coder-next/
#     aime-hybrid-qwen3-coder-next-20260311_204808.json
#   livemathbench/test/qwen3-coder-next/
#     livemathbench-test-qwen3-coder-next-20260312_002223.json
#   gpqa/test/qwen3-coder-next/
#     gpqa-test-qwen3-coder-next-20260312_123649.json
#   mmlupro/test_3000/qwen3-coder-next/
#     mmlupro-test_3000-qwen3-coder-next-20260312_171632.json
```

### 验证 JSON 文件完整性

使用 Python 脚本验证：

```python
import json
import os

results = [
    'results/bench/aime/hybrid/qwen3-coder-next/aime-hybrid-qwen3-coder-next-*.json',
    'results/bench/livemathbench/test/qwen3-coder-next/livemathbench-test-qwen3-coder-next-*.json',
    'results/bench/gpqa/test/qwen3-coder-next/gpqa-test-qwen3-coder-next-*.json',
    'results/bench/mmlupro/test_3000/qwen3-coder-next/mmlupro-test_3000-qwen3-coder-next-*.json',
]

for pattern in results:
    import glob
    files = glob.glob(pattern)
    if not files:
        print(f"❌ Missing: {pattern}")
        continue
    
    filepath = files[0]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证必需字段
        required = ['performance', 'counts', 'records', 'model_name', 'dataset_name']
        missing = [f for f in required if f not in data]
        
        if missing:
            print(f"❌ {filepath}: Missing fields {missing}")
        elif not (0.0 <= data['performance'] <= 1.0):
            print(f"❌ {filepath}: Invalid performance {data['performance']}")
        elif data['counts'] != len(data['records']):
            print(f"❌ {filepath}: Counts mismatch ({data['counts']} != {len(data['records'])})")
        else:
            print(f"✅ {filepath}: Valid ({data['counts']} records, {data['performance']:.2%} accuracy)")
    
    except Exception as e:
        print(f"❌ {filepath}: Error - {e}")
```

### 预期结果（qwen3-coder-next 模型）

| 数据集 | 记录数 | 准确率 | 成本 | 文件大小 |
|--------|--------|--------|------|----------|
| AIME | 60 | 86.67% | $0.0046 | ~1.2 MB |
| LiveMathBench | 121 | 79.34% | $0.0056 | ~1.4 MB |
| GPQA | 198 | 73.23% | $0.0079 | ~2.8 MB |
| MMLU-Pro | 3000 | 82.37% | $0.0579 | ~50 MB |
| **总计** | **3379** | **80.40%** | **$0.076** | **~55 MB** |

---

## 常见问题

### Q1: 收集过程中断怎么办？

**A**: 数据收集工具支持断点续传：

1. 检查哪些数据集已完成：
   ```bash
   ls results/bench/*/test/<model_name>/*.json
   ```

2. 修改配置文件，只保留未完成的数据集

3. 重新运行收集器（`overwrite: false` 会跳过已存在的结果）

### Q2: API 一直超时怎么办？

**A**: 降低并发数：

```yaml
run:
  concurrency: 2  # 从 4 降到 2
```

或者增加超时时间：

```yaml
models:
  - timeout: 600  # 从 300 秒增加到 600 秒
```

### Q3: 缓存不工作，每次都重新调用 API

**警告信息**:
```
'cryptography' package is required for sha256_password or caching_sha2_password auth methods
```

**原因**: MySQL 缓存需要 `cryptography` 包。

**解决方案**:

```bash
pip install cryptography
```

**影响**: 不安装此包不会阻止收集，但会禁用缓存功能，导致成本增加。

### Q4: 如何估算收集成本？

**公式**:

```
总成本 = (输入 tokens × 输入单价 + 输出 tokens × 输出单价) / 1,000,000
```

**参考单价** (SuperRouter):
- 输入: $0.01 / 百万 tokens
- 输出: $0.01 / 百万 tokens

**实际成本** (qwen3-coder-next):
- AIME (60 条): $0.0046
- LiveMathBench (121 条): $0.0056
- GPQA (198 条): $0.0079
- MMLU-Pro (3000 条): $0.0579
- **平均每条**: $0.000022

### Q5: 收集到一半想停止怎么办？

**方式 1**: 前台运行时按 `Ctrl+C`

**方式 2**: 后台运行时找到进程并终止：

```bash
# Windows:
wmic process where "CommandLine like '%data_collector%'" get ProcessId
wmic process where "ProcessId=<PID>" delete

# Linux/macOS:
ps aux | grep data_collector
kill <PID>
```

**注意**: 已完成的数据集结果会保留，下次运行时自动跳过。

---

## 性能优化

### 并发数选择

| 并发数 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| 1 | 稳定，不会触发限流 | 非常慢 | API 极不稳定时 |
| 4 | 平衡速度和稳定性 | 偶尔超时 | **推荐** |
| 8 | 速度快 | 频繁超时和限流 | API 稳定时 |

### 重试策略

工具已内置重试机制（最多 10 次），无需手动配置。如需调整：

```python
# 在 generators/generator.py 中
@retry(
    stop=stop_after_attempt(10),  # 修改这里
    wait=wait_exponential(multiplier=1, min=4, max=60),
    ...
)
```

### 日志级别

调试时可以增加日志详细程度：

```yaml
run:
  log_level: DEBUG  # INFO (默认) / DEBUG / WARNING / ERROR
```

---

## 下一步

数据收集完成后，可以执行以下步骤：

1. **运行 Adaptor**: 将 bench 结果转换为训练/测试数据集
   ```bash
   python -m baselines.AvengersPro.adaptor
   ```

2. **训练路由器**: 使用 simple_cluster_router 训练模型
   ```bash
   python -m baselines.AvengersPro.simple_cluster_router train
   ```

3. **生成分析报告**: 评估路由器性能和成本节省
   ```bash
   python -m baselines.AvengersPro.simple_cluster_router analyze
   ```

---

## 附录

### 完整的环境变量列表

```bash
# API 密钥（.env 文件中配置）
INTEL_SUPERROUTER_API_KEY=sk-xxx

# MySQL 配置（.env 文件中配置）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password

# 代理绕过（每次终端都需要设置）
NO_PROXY=superrouter.intel.com,localhost,127.0.0.1
no_proxy=superrouter.intel.com,localhost,127.0.0.1
```

### Git 提交记录

所有修复已提交到本地 Git 仓库（未推送到远程）：

```bash
# 查看提交历史
git log --oneline -3

# 输出:
# e9591ab fix(gpqa): add UTF-8 encoding for Windows compatibility
# 5e1de26 perf(config): reduce concurrency to 4 for SuperRouter stability
# 5ce6aed fix(generators): add SSL verify=False for SuperRouter
```

### 相关文件路径

```
D:/router/LLMRouterBench/
├── config/
│   ├── data_collector_local_vs_cloud.yaml    # 原始配置（2 模型 × 5 数据集）
│   └── data_collector_remaining.yaml         # 临时配置（1 模型 × 2 数据集）
├── generators/
│   └── generator.py                          # 已修复 SSL（line 68）
├── evaluation/
│   └── base_evaluator.py                    # 已修复编码（line 30）
├── data/
│   ├── AIME/                                 # 60 条记录
│   ├── LiveMathBench/                        # 121 条记录
│   ├── GPQA/                                 # 198 条记录
│   ├── MMLUPro/                              # 3000 条记录
│   └── LiveCodeBench/                        # 缺少数据文件
├── results/
│   └── bench/                                # 收集结果存储位置
└── .sisyphus/
    └── evidence/                             # 验证证据文件
```

---

