# 迁移攻击可视化功能指南

## 🎯 概述

迁移测试系统现在支持7种不同类型的可视化图表，深入分析使用Meta-Predictor Framework在模型间的攻击迁移效果。所有图表都以高分辨率(300 DPI)保存。

## 📊 支持的图表类型

### 1. 攻击成功率热力图 (ASR Heatmap)
- **文件名**: `heatmap_asr.png`
- **描述**: 展示不同源模型到目标模型的攻击成功率
- **颜色**: 红色表示高成功率，绿色表示低成功率
- **用途**: 快速识别最佳和最差的迁移组合

### 2. 攻击危害评分热力图 (AHS Heatmap)
- **文件名**: `heatmap_ahs.png`
- **描述**: 展示攻击的危害程度评分
- **颜色**: 红色表示高危害，白色表示低危害
- **用途**: 评估攻击的潜在风险

### 3. 综合分析图表 (Comprehensive Analysis)
- **文件名**: `comprehensive_analysis.png`
- **布局**: 2x2子图包含：
  - ASR对比柱状图
  - AHS对比柱状图
  - 查询次数分布箱线图
  - 时间效率散点图
- **用途**: 全方位对比分析

### 4. 3D效果分析图 (3D Analysis)
- **文件名**: `3d_analysis.png`
- **描述**: 三维空间展示源模型、目标模型和ASR的关系
- **轴**: X轴(源模型), Y轴(目标模型), Z轴(ASR)
- **用途**: 直观展示三维关系

### 5. 多维度雷达图 (Radar Chart)
- **文件名**: `radar_analysis.png`
- **维度**: ASR, AHS, 平均查询次数, 平均时间
- **描述**: 展示不同源模型的综合性能
- **用途**: 性能平衡分析

### 6. 相关性矩阵 (Correlation Matrix)
- **文件名**: `correlation_matrix.png`
- **描述**: 显示各指标间的相关性
- **颜色**: 蓝色正相关，红色负相关
- **用途**: 发现指标间的关系

### 7. 效率分析图 (Efficiency Analysis)
- **文件名**: `efficiency_analysis.png`
- **包含**: 
  - 时间-效果关系图
  - 查询-效果关系图
- **用途**: 优化攻击策略

## 🚀 使用方法

### 交互式生成
```bash
python transfer_test.py
# 选择菜单选项 5: 生成可视化
```

### 命令行批量生成
```bash
# 运行测试后自动生成
python transfer_test.py --source llama2-7b --target mistral-7b --batch-file batch_examples.txt
```

### 演示生成
```bash
# 使用示例数据快速查看效果
python simple_demo.py
```

## 📁 输出目录结构

```
outputs/
├── results_YYYYMMDD_HHMMSS/
│   ├── results.json          # 详细结果
│   ├── results.csv           # 表格数据
│   ├── heatmap_asr.png       # ASR热力图
│   ├── heatmap_ahs.png       # AHS热力图
│   ├── comprehensive_analysis.png
│   ├── 3d_analysis.png
│   ├── radar_analysis.png
│   ├── correlation_matrix.png
│   └── efficiency_analysis.png
└── ...
```

## 🎨 图表特点

### 高质量输出
- **分辨率**: 300 DPI
- **格式**: PNG
- **尺寸**: 适合A4打印

### 易读性设计
- **标签**: 清晰的轴标签和图例
- **字体**: 支持中英文显示
- **布局**: 自动调整避免重叠

### 可定制性
- **颜色映射**: 可修改colormap
- **尺寸**: 可调整图表大小
- **标签**: 可自定义标签文本

## 📈 数据分析建议

### 关键观察点
1. **对角线模式**: 检查模型自身的鲁棒性
2. **对称性**: 分析A→B和B→A的差异
3. **聚类**: 识别相似迁移效果的模型组
4. **异常值**: 发现意外的迁移效果

### 报告撰写建议
- 使用热力图展示整体趋势
- 用柱状图突出最佳/最差组合
- 通过相关性分析解释结果
- 结合雷达图展示综合性能

## 🔧 故障排除

### 常见问题
1. **字体问题**: 确保matplotlib支持所需字体
2. **内存问题**: 大数据集可调整图表尺寸
3. **显示问题**: 使用`plt.show()`预览

### 环境要求
```bash
pip install matplotlib seaborn pandas numpy
```

## 💡 最佳实践

1. **批量测试**: 先运行完整测试再生成图表
2. **对比分析**: 同时查看多个图表获得全面视角
3. **保存原始**: 保留JSON数据便于后续重新分析
4. **版本控制**: 为不同实验创建独立的输出目录

## 📋 快速开始清单

- [ ] 运行示例: `python simple_demo.py`
- [ ] 查看输出: 检查 `demo_outputs/` 目录
- [ ] 运行实际测试: 使用 `transfer_test.py`
- [ ] 分析结果: 查看生成的图表和报告