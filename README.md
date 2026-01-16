<div align="center">

# 大模型相似性指标计算工具

此工具用于计算不同大模型之间的相似性指标，支持对3*5个大模型的组合分别计算15组指标，并将每一类别的指标结果统一记录在一起。

## 工具概述

本工具包含两个主要的Python脚本：

1. **model_similarity_analysis.py** - 完整版本，实现了所有三类相似性指标的计算，提供详细的可视化和报告生成功能
2. **test_model_similarity.py** - 测试脚本，用于验证主要功能是否正常工作

## 相似性指标类别

根据文档，相似性指标分为以下三类：

### 1. 输出分布相似性指标
- KL散度 (KL Divergence)：衡量一个模型的输出分布相对于另一个模型的差异程度
- Jensen-Shannon散度 (Jensen-Shannon Divergence, JSD)：KL的对称、平滑版本，取值范围[0,1]
- Earth Mover's Distance (EMD)：将一个分布变成另一个分布的最小"搬运成本"
- Logits余弦相似度：测量输出向量夹角，简单且与尺度无关
- RBO (Rank-Based Overlap)：测量排名前k个token的重合程度

### 2. 表征空间相似性指标
- Centered Kernel Alignment (CKA)：比较两组嵌入表示之间的相似性
- SVCCA (Singular Value Canonical Correlation Analysis)：找到两个表示空间中高度相关的子空间
- PWCCA (Weighted SVCCA)：对SVCCA加权，强调重要方向
- RSA (Representational Similarity Analysis)：比较嵌入空间中样本两两之间的距离模式

### 3. 行为/功能相似性指标
- 任务一致率：两个模型在相同输入下top-1输出相同的比例
- Pass@k一致率：两个模型在top-k输出中同时包含正确/目标答案的概率
- 对抗迁移率：针对模型A的jailbreak提示在模型B上也成功的比例
- 语义相似性：用句向量计算两个模型输出的语义余弦相似性

## 支持的模型组合

工具支持计算以下3个源模型与5个目标模型的15种组合：

**源模型：**
- llama2-7b
- bert-large
- roberta-large

**目标模型：**
- mistral-7b
- vicuna-7b
- guanaco-7b
- starling-7b
- chatgpt-3.5

## 使用方法

### 环境要求

- Python 3.6+（推荐3.8+）
- 必要的Python库：numpy, matplotlib

可以使用以下命令安装必要的库：

```bash
pip install numpy matplotlib
```

### 运行脚本

#### 运行完整的15组模型对计算

```bash
python model_similarity_analysis.py
```

#### 运行测试脚本

如果您想先测试功能而不运行完整计算：

```bash
python test_model_similarity.py
```

## 输出结果

### 完整计算输出

运行完整版本后，将在当前目录下创建一个名为`similarity_analysis_results`的文件夹，包含以下内容：

1. **JSON格式结果文件**：包含所有指标的详细计算结果
2. **分类结果文件**：按指标类别（输出分布、表征空间、行为/功能）分别保存结果，每一类别的指标结果统一记录在一起
3. **可视化图表**：各类指标的热力图，直观展示不同模型对之间的相似性
4. **总结报告**：包含计算时间、模型信息、各类指标的统计摘要等

### 测试脚本输出

运行测试脚本后，将在当前目录下创建一个名为`test_similarity_results`的文件夹，包含单对模型的计算结果，用于验证功能是否正常。

## 脚本功能详解

### model_similarity_analysis.py 主要功能

- **数据生成**：生成模拟的模型输出概率分布、logits、隐藏层表示等数据
- **指标计算**：实现所有三类共12种相似性指标的计算
- **结果保存**：将结果保存为JSON格式，按指标类别分别记录
- **可视化**：创建热力图，直观展示计算结果
- **报告生成**：生成详细的总结报告，包含各类指标的统计信息

### test_model_similarity.py 主要功能

- **实例验证**：验证ModelSimilarityAnalysis类能否正确初始化
- **模型列表验证**：检查源模型和目标模型的配置是否正确
- **单对计算测试**：测试计算一对模型的指标功能是否正常
- **结果结构检查**：验证计算结果的数据结构是否符合预期
- **结果保存测试**：测试结果保存功能是否正常

## 自定义配置

### 调整模型列表

如果需要调整源模型或目标模型列表，可以修改脚本中的以下部分：

```python
# 源模型列表
source_models = ['llama2-7b', 'bert-large', 'roberta-large']
# 目标模型列表
target_models = ['mistral-7b', 'vicuna-7b', 'guanaco-7b', 'starling-7b', 'chatgpt-3.5']
```

### 调整输出目录

可以通过修改初始化参数来自定义输出目录：

```python
# 修改输出目录
metrics_calculator = ModelSimilarityAnalysis(output_dir="./custom_results")
```

## 注意事项

1. 本工具生成的是模拟数据，实际应用中应替换为真实模型的输出数据
2. 对于对抗迁移率和语义相似性等指标，在实际应用中需要基于真实的模型输出和评估方法
3. 计算大量模型对时，部分指标（如EMD、SVCCA等）可能会消耗较多计算资源

