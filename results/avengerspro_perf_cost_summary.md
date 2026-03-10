# AvengersPro Router 实验总结（Performance-Cost Setting）

## 一、实验 Setting

**目标**：训练一个 LLM Router，为每条用户 query 自动选择最优模型，在保持高准确率的同时控制推理成本。

### 候选模型池（13 个 flagship models）

| 类别 | 模型 |
|------|------|
| OpenAI | gpt-5, gpt-5-chat |
| Google | gemini-2.5-pro, gemini-2.5-flash |
| Anthropic | claude-sonnet-4 |
| DeepSeek | deepseek-r1-0528, deepseek-v3-0324, deepseek-v3.1-terminus |
| 阿里 | qwen3-235b-a22b-2507, qwen3-235b-a22b-thinking-2507 |
| 其他 | glm-4.6, kimi-k2-0905, intern-s1 |

### 评测数据集（10 个）

AIME, ArenaHard, GPQA, HLE, LiveCodeBench, LiveMathBench, MMLUPro, SimpleQA, SWE-Bench, TAU2

### 数据划分

- 拆分比例：70% 训练 / 30% 测试（seed=42）
- 训练集：8,491 条（原始 8,710 条，剔除 219 条 swe-bench 超长 query）
- 测试集：3,647 条（原始 3,738 条，剔除 92 条 swe-bench 超长 query）
- 剔除原因：swe-bench 部分 query 长达 ~240K tokens，超出 embedding 模型  的 8192 token 上限

### 每条数据结构

一条 query + 13 个模型各自在该 query 上的真实得分（1.0=正确 / 0.0=错误 / 0.5=部分正确，仅 arenahard）

---

## 二、训练方法

**AvengersPro** 采用**基于 K-Means 聚类的路由策略**：

1. **Embedding**：使用 `text-embedding-3-large`（3072 维）将每条 query 编码为向量，L2 归一化
2. **聚类**：对训练集的 query embedding 做 K-Means 聚类（K=16），将语义相似的 query 分到同一个 cluster
3. **学习 cluster 内模型排名**：对每个 cluster，统计该 cluster 内各模型的平均得分，建立模型性能排名表
4. **推理路由**：给定新 query →
   - 计算 query embedding
   - 找到最近的 cluster（top_k=1）
   - 根据该 cluster 的模型排名，选出排名第一的模型（max_router=1）
   - 用 softmax 加权（beta=9.0）融合 cluster 距离作为置信度

### 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_clusters | 16 | K-Means 聚类数 |
| top_k | 1 | 路由时考虑最近 cluster 数 |
| max_router | 1 | 最终选出的模型数 |
| beta | 9.0 | softmax 温度参数 |
| cost_sensitivity | 0.3 | 成本敏感度 |
| performance_weight | 0.7 | 性能权重 |
| embedding_model | text-embedding-3-large | 3072 维 |
| max_tokens | 7500 | query 最大 token 数（留 margin） |
| seed | 42 | 随机种子 |

---

## 三、评估方法

**Router 准确率定义**：Router 为每条 query 选择 1 个模型，以该模型在该 query 上的 ground truth 得分作为正确性。

- **Dataset-Avg**：先算各数据集准确率，再取 10 个数据集等权平均（避免大数据集主导）
- **Sample-Avg**：所有 query 得分直接取平均

**对比基线**：
- **随机选**：在 13 个模型中随机选 1 个的期望准确率
- **最佳单模型**：始终使用表现最好的单个模型
- **Oracle 上界**：每条 query 总是选到得分最高的模型（路由的理论上限）

---

## 四、实验结果

### 4.1 准确率对比

| 数据集 | 问题数 | 随机选 | Router | 最佳单模型 | Oracle上界 |
|--------|--------|--------|--------|-----------|-----------|
| aime | 18 | 64.53% | **83.33%** | 88.89% (glm-4.6) | 88.89% |
| arenahard | 226 | 68.16% | 68.14% | 78.10% (qwen3-thinking) | 99.12% |
| gpqa | 60 | 73.97% | **86.67%** | 88.33% (gpt-5) | 96.67% |
| hle | 647 | 10.94% | **25.81%** | 25.93% (gpt-5) | 50.85% |
| livecodebench | 317 | 63.82% | **84.54%** | 84.54% (gpt-5) | 91.80% |
| livemathbench | 37 | 62.58% | **72.97%** | 78.38% (gpt-5) | 81.08% |
| mmlupro | 900 | 83.02% | **87.56%** | 87.67% (gemini-2.5-pro) | 94.22% |
| simpleqa | 1298 | 33.86% | **55.78%** | 54.01% (gemini-2.5-pro) | 85.82% |
| swe-bench | 60 | 27.56% | **43.33%** | 36.00% (gemini-2.5-pro) | 73.33% |
| tau2 | 84 | 38.10% | **69.05%** | 69.05% (gpt-5) | 94.05% |
| **Dataset-Avg** | **3647** | **52.65%** | **67.72%** | **65.11%** (gpt-5) | **85.58%** |

### 4.2 关键发现

1. **Router（67.72%）超越了最佳单模型 baseline gpt-5（65.11%）**，提升 +2.61 个百分点

2. **Router 超越所有单模型的数据集**：
   - **swe-bench**：43.33% vs 36.00%（**+7.33%**，提升最大）
   - **simpleqa**：55.78% vs 54.01%（+1.77%）

3. **Router 与最佳单模型持平的数据集**：
   - livecodebench（84.54% ≈ 84.54%）
   - tau2（69.05% ≈ 69.05%）
   - hle（25.81% ≈ 25.93%）
   - mmlupro（87.56% ≈ 87.67%）

4. **Router 明显落后的数据集**：
   - arenahard（68.14% vs 78.10%，**-9.96%**）

5. **相比随机选提升 +15.06 个百分点**，说明聚类路由有效学到了 query-模型匹配模式

6. **与 Oracle 上界（85.58%）的差距为 17.86 个百分点**，仍有优化空间

### 4.3 模型选择分布

Router 在 13 个候选中只实际使用了 4 个模型：

| 模型 | 选择次数 | 占比 | 花费 |
|------|---------|------|------|
| gpt-5 | 1,928 | 52.9% | 7.90 |
| gemini-2.5-pro | 1,155 | 31.7% | 1.01 |
| qwen3-235b-a22b-2507 | 387 | 10.6% | /usr/bin/bash.01 |
| claude-sonnet-4 | 177 | 4.9% | .57 |

### 4.4 成本分析

| 指标 | Router | gpt-5 (最佳baseline) | deepseek-v3-0324 (最便宜) |
|------|--------|---------------------|--------------------------|
| Dataset-Avg 准确率 | **67.72%** | 65.11% | 42.60% |
| 总花费 | 23.49 | 19.16 | .14 |
| 单条 query 成本 | /usr/bin/bash.034 | /usr/bin/bash.033 | /usr/bin/bash.001 |

Router 在准确率上胜出 +2.61%，但成本与 gpt-5 基本持平（23.49 vs 19.16），因为 Router 有 52.9% 的 query 选择了 gpt-5。

---

## 五、复现命令



### 注意事项

- 训练数据使用了 （剔除了超长 swe-bench query），非原始  / 
-  从默认 30000 调整为 7500， 从 32 调整为 8，以适配 embedding API 限制
- 配置文件：
- 输出文件：
