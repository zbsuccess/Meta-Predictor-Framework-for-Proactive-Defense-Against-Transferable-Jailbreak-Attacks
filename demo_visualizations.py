#!/usr/bin/env python3
"""
可视化功能演示脚本
生成示例数据并展示所有图表类型
"""

import os
import sys
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体和样式 - 使用英文字体避免中文问题
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
sns.set_style("whitegrid")

class VisualizationDemo:
    def __init__(self):
        self.output_dir = "demo_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_sample_data(self):
        """生成示例数据"""
        source_models = ['bert-large', 'roberta-large', 'llama2-7b']
        target_models = ['mistral-7b', 'vicuna-7b', 'guanaco-7b', 'starling-7b', 'chatgpt-3.5']
        
        results = []
        for source in source_models:
            for target in target_models:
                result = {
                    'source_model': source,
                    'target_model': target,
                    'ASR': random.uniform(0.1, 0.9),
                    'AHS': random.uniform(0.2, 0.8),
                    'Average Queries': random.randint(50, 200),
                    'Average Time': random.uniform(5, 30),
                    'status': 'success'
                }
                results.append(result)
        
        return results
    
    def create_demo_charts(self):
        """创建演示图表"""
        print("🎨 生成演示数据...")
        results = self.generate_sample_data()
        df = pd.DataFrame(results)
        
        print(f"📊 生成了 {len(results)} 组测试数据")
        
        # 1. 热力图 - ASR
        plt.figure(figsize=(12, 8))
        pivot_asr = df.pivot(index='source_model', columns='target_model', values='ASR')
        sns.heatmap(pivot_asr, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                   cbar_kws={'label': '攻击成功率'}, vmin=0, vmax=1)
        plt.title("攻击成功率 (ASR) 热力图", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_heatmap_asr.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 热力图 - AHS
        plt.figure(figsize=(12, 8))
        pivot_ahs = df.pivot(index='source_model', columns='target_model', values='AHS')
        sns.heatmap(pivot_ahs, annot=True, fmt=".3f", cmap="Reds", 
                   cbar_kws={'label': '攻击危害评分'}, vmin=0, vmax=1)
        plt.title("攻击危害评分 (AHS) 热力图", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_heatmap_ahs.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 综合对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('迁移攻击效果综合分析', fontsize=16)
        
        # ASR 柱状图
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='source_model', y='ASR', hue='target_model', ax=ax1)
        ax1.set_title('各模型组合的ASR对比')
        ax1.tick_params(axis='x', rotation=45)
        
        # AHS 柱状图
        ax2 = axes[0, 1]
        sns.barplot(data=df, x='source_model', y='AHS', hue='target_model', ax=ax2)
        ax2.set_title('各模型组合的AHS对比')
        ax2.tick_params(axis='x', rotation=45)
        
        # 查询次数箱线图
        ax3 = axes[1, 0]
        sns.boxplot(data=df, x='target_model', y='Average Queries', ax=ax3)
        ax3.set_title('目标模型查询次数分布')
        ax3.tick_params(axis='x', rotation=45)
        
        # 时间效率散点图
        ax4 = axes[1, 1]
        sns.scatterplot(data=df, x='Average Time', y='ASR', 
                       hue='source_model', size='Average Queries', ax=ax4)
        ax4.set_title('时间效率 vs 攻击成功率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_comprehensive_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3D 效果分析
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建数字映射
        source_map = {s: i for i, s in enumerate(df['source_model'].unique())}
        target_map = {t: i for i, t in enumerate(df['target_model'].unique())}
        
        x = [source_map[s] for s in df['source_model']]
        y = [target_map[t] for t in df['target_model']]
        z = df['ASR']
        
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=100)
        ax.set_xlabel('Source Model')
        ax.set_ylabel('Target Model')
        ax.set_zlabel('ASR')
        ax.set_title('3D Transfer Effect Analysis')
        
        # 设置刻度标签
        ax.set_xticks(list(source_map.values()))
        ax.set_xticklabels(list(source_map.keys()), rotation=45)
        ax.set_yticks(list(target_map.values()))
        ax.set_yticklabels(list(target_map.keys()), rotation=45)
        
        plt.colorbar(scatter)
        plt.savefig(os.path.join(self.output_dir, "demo_3d_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 雷达图 - 多维度分析
        metrics = ['ASR', 'AHS', 'Average Queries', 'Average Time']
        
        # 根据源模型数量调整子图
        n_sources = len(df['source_model'].unique())
        fig, axes = plt.subplots(1, min(n_sources, 3), figsize=(16, 6), 
                                subplot_kw=dict(projection='polar'))
        if n_sources == 1:
            axes = [axes]  # 确保axes是可迭代的
        
        fig.suptitle('Multi-dimensional Performance Radar Chart', fontsize=16)
        
        # 按源模型分组
        for idx, (source, group) in enumerate(df.groupby('source_model')):
            if idx >= len(axes):
                break
                
            values = group[metrics].mean().values
            values = (values - values.min()) / (values.max() - values.min() + 1e-8)  # 归一化
            
            angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
            values = np.concatenate((values, [values[0]]))  # 闭合
            angles = np.concatenate((angles, [angles[0]]))
            
            ax = axes[idx]
            ax.plot(angles, values, 'o-', linewidth=2, label=source)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title(f'{source} Performance Radar')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_radar_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. 相关性矩阵
        plt.figure(figsize=(10, 8))
        corr_matrix = df[metrics].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_correlation_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. 效率分析
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        for source in df['source_model'].unique():
            source_data = df[df['source_model'] == source]
            plt.plot(source_data['Average Time'], source_data['ASR'], 
                    marker='o', label=source, linewidth=2)
        plt.xlabel('平均时间 (秒)')
        plt.ylabel('ASR')
        plt.title('时间-效果关系')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for target in df['target_model'].unique():
            target_data = df[df['target_model'] == target]
            plt.scatter(target_data['Average Queries'], target_data['ASR'], 
                       label=target, s=100, alpha=0.7)
        plt.xlabel('平均查询次数')
        plt.ylabel('ASR')
        plt.title('查询-效果关系')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "demo_efficiency_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存示例数据
        with open(os.path.join(self.output_dir, "demo_results.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'summary': {
                    'total_tests': len(results),
                    'average_asr': df['ASR'].mean(),
                    'average_ahs': df['AHS'].mean(),
                    'generated_at': datetime.now().isoformat()
                }
            }, f, indent=2, ensure_ascii=False)
        
        print("✅ 演示图表生成完成!")
        print(f"📁 文件保存在: {self.output_dir}/")
        
        # 显示生成的文件列表
        files = os.listdir(self.output_dir)
        print("\n📋 生成的文件:")
        for file in sorted(files):
            print(f"  - {file}")

if __name__ == "__main__":
    demo = VisualizationDemo()
    demo.create_demo_charts()