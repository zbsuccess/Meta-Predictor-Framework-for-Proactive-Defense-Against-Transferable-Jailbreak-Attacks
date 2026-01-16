# 迁移攻击测试系统 - 快速使用指南

## 核心文件
- `transfer_test.py` - 主程序（集成所有功能）
- `batch_examples.txt` - 批量测试示例

## 支持的模型

### 源模型 (A - 攻击生成)
- `llama2-7b` - meta-llama/Llama-2-7b-chat-hf
- `bert-large` - bert-large-uncased  
- `roberta-large` - FacebookAI/roberta-large

### 目标模型 (B - 被攻击)
- `mistral-7b` - mistralai/Mistral-7B-Instruct-v0.2
- `vicuna-7b` - lmsys/vicuna-7b-v1.5
- `guanaco-7b` - TheBloke/guanaco-7B-HF
- `starling-7b` - berkeley-nest/Starling-LM-7B-alpha
- `chatgpt-3.5` - Dac120/Chat-GPT-3.5

## 快速开始

### 1. 交互式运行（推荐）
```bash
python transfer_test.py --interactive
```

在交互模式中，你可以选择：
1. 查看可用模型
2. 运行单个测试
3. 运行批量测试
4. 查看历史结果
5. 生成多样化可视化
6. 显示总结报告
7. 退出

### 2. 单个测试
```bash
python transfer_test.py --source bert-large --target mistral-7b
```

### 3. 批量测试
```bash
python transfer_test.py --batch batch_examples.txt
```

### 4. 自定义参数
```bash
python transfer_test.py --source llama2-7b --target vicuna-7b --objective ASR+GPT
```

## 输出结果

- `transfer_results/` 目录下包含：
  - JSON格式详细结果
  - CSV格式汇总表格
  - 迁移效果热力图

### 可视化功能
- **多样化图表**: 7种不同类型的可视化分析
  - 攻击成功率热力图 (ASR)
  - 攻击危害评分热力图 (AHS)
  - 综合分析图表 (2x2子图)
  - 3D效果分析图
  - 多维度雷达图
  - 指标相关性矩阵
  - 时间效率分析图
- **交互式图表**: 支持缩放、保存
- **高质量输出**: 300 DPI PNG格式

## 关键指标
- **ASR**: 攻击成功率
- **AHS**: 攻击危害评分
- **Average Queries**: 平均查询次数
- **Average Time**: 平均运行时间