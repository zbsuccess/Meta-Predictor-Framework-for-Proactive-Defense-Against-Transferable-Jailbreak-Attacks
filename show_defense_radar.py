#!/usr/bin/env python3
"""
展示所有目标模型的防御雷达图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_data():
    """创建示例数据"""
    data = []
    sources = ['bert-large', 'llama2-7b', 'roberta-base', 't5-base']
    targets = ['mistral-7b', 'vicuna-7b', 'llama2-13b', 'falcon-7b']
    
    for source in sources:
        for target in targets:
            data.append({
                'source_model': source,
                'target_model': target,
                'ASR': np.random.uniform(0.4, 0.9),
                'AHS': np.random.uniform(0.3, 0.8),
                'Average Queries': np.random.randint(70, 180),
                'Average Time': np.random.uniform(10, 28)
            })
    
    return pd.DataFrame(data)

def create_defense_radar_all_targets(df, output_dir="defense_output"):
    """为所有目标模型创建防御能力雷达图"""
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有目标模型
    target_models = sorted(df['target_model'].unique())
    n_targets = len(target_models)
    
    print(f"发现 {n_targets} 个目标模型: {target_models}")
    
    # 创建大图
    fig, axes = plt.subplots(2, 2, 
                           figsize=(16, 12), 
                           subplot_kw=dict(projection='polar'))
    
    # 调整布局
    fig.suptitle('Target Model Defense Capability Radar', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    # 定义防御指标（英文）
    metric_names = {
        'ASR Defense': 'ASR Defense',
        'AHS Defense': 'AHS Defense', 
        'Query Efficiency': 'Query Efficiency',
        'Time Efficiency': 'Time Efficiency'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, target in enumerate(target_models):
        if idx >= len(axes):
            break
            
        target_data = df[df['target_model'] == target]
        
        # 计算防御指标
        defense_metrics = {
            'ASR Defense': 1 - target_data['ASR'].mean(),
            'AHS Defense': 1 - target_data['AHS'].mean(),
            'Query Efficiency': max(0, 1 - target_data['Average Queries'].mean()/200),
            'Time Efficiency': max(0, 1 - target_data['Average Time'].mean()/30)
        }
        
        metrics = list(defense_metrics.values())
        labels = list(metric_names.values())
        
        print(f"\n目标模型: {target}")
        for metric, value in defense_metrics.items():
            print(f"  {metric_names[metric]}: {value:.3f}")
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        metrics = np.concatenate((metrics, [metrics[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax = axes[idx]
        
        # 绘制雷达图
        ax.plot(angles, metrics, 'o-', linewidth=3, color=colors[idx % len(colors)], 
                label=target, markersize=8)
        ax.fill(angles, metrics, alpha=0.25, color=colors[idx % len(colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        
        # 设置标题
        ax.set_title(f'{target}', fontsize=14, fontweight='bold', pad=30)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
    
    # 隐藏多余的子图
    for idx in range(n_targets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, "all_targets_defense_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 同时创建单个模型的详细图
    for target in target_models:
        create_single_target_radar(df, target, output_dir)
    
    return output_path

def create_single_target_radar(df, target, output_dir):
    """为单个目标模型创建详细雷达图"""
    
    target_data = df[df['target_model'] == target]
    
    # 计算防御指标
    defense_metrics = {
        'ASR Defense': 1 - target_data['ASR'].mean(),
        'AHS Defense': 1 - target_data['AHS'].mean(),
        'Query Efficiency': max(0, 1 - target_data['Average Queries'].mean()/200),
        'Time Efficiency': max(0, 1 - target_data['Average Time'].mean()/30)
    }
    
    metrics = list(defense_metrics.values())
    labels = list(defense_metrics.keys())
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 设置样式
    ax.set_facecolor('#f8f9fa')
    
    # 创建雷达图
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    metrics = np.concatenate((metrics, [metrics[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    # 绘制
    ax.plot(angles, metrics, 'o-', linewidth=3, color='navy', markersize=8)
    ax.fill(angles, metrics, alpha=0.3, color='lightblue')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    
    # 设置标题
    ax.set_title(f'{target} Defense Capability Analysis', 
                 fontsize=16, fontweight='bold', pad=30)
    
    # 添加数值标签
    for angle, value in zip(angles[:-1], metrics[:-1]):
        ax.text(angle, value + 0.05, f'{value:.2f}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加网格和样式
    ax.grid(True, alpha=0.4)
    
    # 保存
    output_path = os.path.join(output_dir, f"{target}_defense_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def main():
    """主函数"""
    print("🎯 开始生成所有目标模型的防御雷达图...")
    
    # 创建示例数据
    df = create_sample_data()
    print("✅ 数据创建完成")
    print(f"数据包含 {len(df)} 条记录")
    
    # 创建输出目录
    output_dir = "all_targets_defense_radar"
    
    # 生成所有雷达图
    combined_path = create_defense_radar_all_targets(df, output_dir)
    
    print(f"\n🎉 所有目标模型的防御雷达图生成完成！")
    print(f"📁 输出目录: {output_dir}")
    
    # 列出所有生成的文件
    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print("\n📋 生成的文件:")
    for f in sorted(files):
        print(f"  ✅ {f}")

if __name__ == "__main__":
    main()