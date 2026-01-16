#!/usr/bin/env python3
"""
简化版可视化演示脚本
"""

import os
import sys
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_demo_data():
    """创建演示数据"""
    source_models = ['bert-large', 'roberta-large', 'llama2-7b']
    target_models = ['mistral-7b', 'vicuna-7b', 'guanaco-7b', 'starling-7b', 'chatgpt-3.5']
    
    results = []
    for source in source_models:
        for target in target_models:
            result = {
                'source_model': source,
                'target_model': target,
                'ASR': round(random.uniform(0.1, 0.9), 3),
                'AHS': round(random.uniform(0.2, 0.8), 3),
                'Average Queries': random.randint(50, 200),
                'Average Time': round(random.uniform(5, 30), 1)
            }
            results.append(result)
    return results

def create_simple_visualizations():
    """创建简化可视化"""
    print("🎨 生成演示数据...")
    results = create_demo_data()
    df = pd.DataFrame(results)
    
    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 热力图
    plt.figure(figsize=(10, 6))
    pivot = df.pivot(index='source_model', columns='target_model', values='ASR')
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                cbar_kws={'label': 'Attack Success Rate'})
    plt.title('Attack Success Rate (ASR) Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_asr.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comprehensive Transfer Attack Analysis', fontsize=16)
    
    # ASR 柱状图
    sns.barplot(data=df, x='source_model', y='ASR', hue='target_model', ax=axes[0,0])
    axes[0,0].set_title('ASR Comparison by Model Pairs')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # AHS 柱状图
    sns.barplot(data=df, x='source_model', y='AHS', hue='target_model', ax=axes[0,1])
    axes[0,1].set_title('AHS Comparison by Model Pairs')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 查询次数箱线图
    sns.boxplot(data=df, x='target_model', y='Average Queries', ax=axes[1,0])
    axes[1,0].set_title('Query Count Distribution')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 时间效率散点图
    sns.scatterplot(data=df, x='Average Time', y='ASR', 
                   hue='source_model', size='Average Queries', ax=axes[1,1])
    axes[1,1].set_title('Time Efficiency vs ASR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comprehensive_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 相关性矩阵
    plt.figure(figsize=(8, 6))
    metrics = ['ASR', 'AHS', 'Average Queries', 'Average Time']
    corr_matrix = df[metrics].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title('Metrics Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 效率分析
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for source in df['source_model'].unique():
        data = df[df['source_model'] == source]
        plt.scatter(data['Average Time'], data['ASR'], label=source, s=100)
    plt.xlabel('Average Time (seconds)')
    plt.ylabel('ASR')
    plt.title('Time vs Attack Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for target in df['target_model'].unique():
        data = df[df['target_model'] == target]
        plt.scatter(data['Average Queries'], data['ASR'], label=target, s=100)
    plt.xlabel('Average Queries')
    plt.ylabel('ASR')
    plt.title('Queries vs Attack Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    with open(os.path.join(output_dir, "sample_data.json"), 'w') as f:
        json.dump({
            'data': results,
            'summary': {
                'total_tests': len(results),
                'average_asr': round(df['ASR'].mean(), 3),
                'average_ahs': round(df['AHS'].mean(), 3)
            }
        }, f, indent=2)
    
    print("✅ 演示图表生成完成!")
    print(f"📁 文件保存在: {output_dir}/")
    
    # 显示文件列表
    files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print("\n📋 生成的图表:")
    for file in sorted(files):
        print(f"  - {file}")

if __name__ == "__main__":
    create_simple_visualizations()