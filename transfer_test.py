#!/usr/bin/env python3


import os
import json
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from datetime import datetime

# 精简的模型配置
MODELS = {
    # 源模型 (A模型 - 用于生成攻击)
    'llama2-7b': {
        'type': 'clm',
        'path': 'meta-llama/Llama-2-7b-chat-hf',
        'description': 'Llama 2 7B Chat'
    },
    'bert-large': {
        'type': 'mlm', 
        'path': 'bert-large-uncased',
        'description': 'BERT Large'
    },
    'roberta-large': {
        'type': 'mlm',
        'path': 'FacebookAI/roberta-large', 
        'description': 'RoBERTa Large'
    },
    
    # 目标模型 (B模型 - 被攻击的模型)
    'mistral-7b': {
        'type': 'target',
        'path': 'mistralai/Mistral-7B-Instruct-v0.2',
        'description': 'Mistral 7B Instruct'
    },
    'vicuna-7b': {
        'type': 'target',
        'path': 'lmsys/vicuna-7b-v1.5',
        'description': 'Vicuna 7B'
    },
    'guanaco-7b': {
        'type': 'target',
        'path': 'TheBloke/guanaco-7B-HF',
        'description': 'Guanaco 7B'
    },
    'starling-7b': {
        'type': 'target',
        'path': 'berkeley-nest/Starling-LM-7B-alpha',
        'description': 'Starling 7B'
    },
    'chatgpt-3.5': {
        'type': 'target',
        'path': 'Dac120/Chat-GPT-3.5',
        'description': 'ChatGPT 3.5'
    }
}

class TransferTester:
    def __init__(self, output_dir="./transfer_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
    
    def run_single_transfer(self, source_model, target_model, objective="ASR"):
        """运行单个迁移测试"""
        
        # 获取模型配置
        source_config = MODELS[source_model]
        target_config = MODELS[target_model]
        
        # 确定测试脚本
        script = "PiF_MLM.py" if source_config['type'] == 'mlm' else "PiF_CLM.py"
        
        # 生成测试名称
        test_name = f"{source_model}_to_{target_model}_{objective}"
        output_subdir = os.path.join(self.output_dir, test_name)
        
        # 构建命令
        cmd = [
            "python", script,
            "--gen_model_path", source_config['path'],
            "--tgt_model_path", target_config['path'],
            "--opt_objective", objective,
            "--output_dir", output_subdir,
            "--output_file", f"{test_name}.json"
        ]
        
        print(f"\n🔄 测试: {source_model} → {target_model}")
        print(f"命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                # 解析结果
                result_data = self.parse_result(output_subdir, f"{test_name}.json")
                result_data.update({
                    'source_model': source_model,
                    'target_model': target_model,
                    'objective': objective,
                    'status': 'success'
                })
                self.results.append(result_data)
                print(f"✅ 完成 - ASR: {result_data['ASR']:.3f}")
                return result_data
            else:
                print(f"❌ 失败: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            print("❌ 超时")
            return {'status': 'timeout'}
    
    def parse_result(self, output_dir, filename):
        """解析测试结果"""
        filepath = os.path.join(output_dir, filename)
        
        if not os.path.exists(filepath):
            return {'ASR': 0, 'AHS': 0, 'Average Queries': 0, 'Average Time': 0}
        
        try:
            with open(filepath) as f:
                lines = f.readlines()
                if lines:
                    summary = json.loads(lines[-1])
                    return {
                        'ASR': summary.get('ASR', 0),
                        'AHS': summary.get('AHS', 0),
                        'Average Queries': summary.get('Average Queries', 0),
                        'Average Time': summary.get('Average Time', 0)
                    }
        except:
            pass
        
        return {'ASR': 0, 'AHS': 0, 'Average Queries': 0, 'Average Time': 0}
    
    def run_batch_transfers(self, pairs, objective="ASR"):
        """批量运行迁移测试"""
        results = []
        for source, target in pairs:
            result = self.run_single_transfer(source, target, objective)
            results.append(result)
            
            # 实时保存
            self.save_results()
        
        return results
    
    def save_results(self, filename=None):
        """保存结果"""
        if not filename:
            filename = f"transfer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 同时保存CSV
        df = pd.DataFrame([r for r in self.results if r.get('status') == 'success'])
        if not df.empty:
            df.to_csv(filepath.replace('.json', '.csv'), index=False)
        
        print(f"💾 结果已保存: {filename}")
    
    def create_visualizations(self):
        """创建多样化可视化图表"""
        if not self.results:
            print("❌ 没有结果数据")
            return
        
        successful = [r for r in self.results if r.get('status') == 'success']
        if not successful:
            print("❌ 没有成功的测试结果")
            return
        
        df = pd.DataFrame(successful)
        
        # 1. 热力图 - ASR
        plt.figure(figsize=(12, 8))
        pivot_asr = df.pivot(index='source_model', columns='target_model', values='ASR')
        sns.heatmap(pivot_asr, annot=True, fmt=".3f", cmap="RdYlGn_r", 
                   cbar_kws={'label': 'Attack Success Rate'}, vmin=0, vmax=1)
        plt.title("Attack Success Rate (ASR) Heatmap", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "heatmap_asr.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 热力图 - AHS
        if 'AHS' in df.columns:
            plt.figure(figsize=(12, 8))
            pivot_ahs = df.pivot(index='source_model', columns='target_model', values='AHS')
            sns.heatmap(pivot_ahs, annot=True, fmt=".3f", cmap="Reds", 
                       cbar_kws={'label': 'Attack Harmfulness Score'}, vmin=0, vmax=1)
            plt.title("Attack Harmfulness Score (AHS) Heatmap", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "heatmap_ahs.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 综合对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Transfer Attack Analysis', fontsize=16)
        
        # ASR 柱状图
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='source_model', y='ASR', hue='target_model', ax=ax1)
        ax1.set_title('各模型组合的ASR对比')
        ax1.tick_params(axis='x', rotation=45)
        
        # AHS 柱状图
        if 'AHS' in df.columns:
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
        plt.savefig(os.path.join(self.output_dir, "comprehensive_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 3D 效果分析
        if len(df) > 4:
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
            plt.savefig(os.path.join(self.output_dir, "3d_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. 雷达图 - 多维度分析
        if len(df) > 0:
            metrics = ['ASR', 'AHS', 'Average Queries', 'Average Time']
            available_metrics = [m for m in metrics if m in df.columns]
            
            if len(available_metrics) >= 3:
                n_sources = len(df['source_model'].unique())
                fig, axes = plt.subplots(1, min(n_sources, 3), figsize=(16, 6), 
                                      subplot_kw=dict(projection='polar'))
                if n_sources == 1:
                    axes = [axes]  # 确保axes是可迭代的
                
                fig.suptitle('Multi-dimensional Performance Radar', fontsize=16)
                
                # 按源模型分组
                for idx, (source, group) in enumerate(df.groupby('source_model')):
                    if idx >= len(axes):
                        break
                    
                    values = group[available_metrics].mean().values
                    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                    
                    angles = [n / float(len(available_metrics)) * 2 * 3.14159 
                             for n in range(len(available_metrics))]
                    values = np.concatenate((values, [values[0]]))
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    ax = axes[idx]
                    ax.plot(angles, values, 'o-', linewidth=2, label=source)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(available_metrics)
                    ax.set_ylim(0, 1)
                    ax.set_title(f'{source} Radar')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "radar_analysis.png"), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 6. 相关性矩阵
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) > 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
            plt.title('Metrics Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. 时间序列分析（如果有多次运行）
        if len(df) > 5:
            df_sorted = df.sort_values('Average Time')
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            for source in df['source_model'].unique():
                source_data = df[df['source_model'] == source]
                plt.plot(source_data['Average Time'], source_data['ASR'], 
                        marker='o', label=source, linewidth=2)
            plt.xlabel('平均时间 (秒)')
            plt.ylabel('ASR')
            plt.title('Time vs Effectiveness')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            for target in df['target_model'].unique():
                target_data = df[df['target_model'] == target]
                plt.scatter(target_data['Average Queries'], target_data['ASR'], 
                           label=target, s=100, alpha=0.7)
            plt.xlabel('Average Queries')
            plt.ylabel('ASR')
            plt.title('Queries vs Effectiveness')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "efficiency_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        print("📊 所有可视化图表已生成完成!")
        print(f"📁 保存位置: {self.output_dir}")
        
        # 生成图表清单
        charts = [
            "heatmap_asr.png - 攻击成功率热力图",
            "heatmap_ahs.png - 攻击危害评分热力图", 
            "comprehensive_analysis.png - 综合分析图表",
            "3d_analysis.png - 3D效果分析",
            "radar_analysis.png - 多维度雷达图",
            "correlation_matrix.png - 相关性矩阵",
            "efficiency_analysis.png - 效率分析图"
        ]
        
        print("\n📋 生成的图表:")
        for chart in charts:
            if os.path.exists(os.path.join(self.output_dir, chart.split(' - ')[0])):
                print(f"  ✅ {chart}")

    def show_summary_report(self):
        """显示总结报告"""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.get('status') == 'success']
        if not successful:
            return
        
        df = pd.DataFrame(successful)
        
        print("\n" + "="*60)
        print("📊 迁移攻击测试总结报告")
        print("="*60)
        
        print(f"总测试数: {len(self.results)}")
        print(f"成功测试: {len(successful)}")
        print(f"成功率: {len(successful)/len(self.results)*100:.1f}%")
        
        if not df.empty:
            print(f"\n📈 关键指标:")
            print(f"  平均ASR: {df['ASR'].mean():.3f}")
            if 'AHS' in df.columns:
                print(f"  平均AHS: {df['AHS'].mean():.3f}")
            print(f"  平均查询: {df['Average Queries'].mean():.1f}")
            print(f"  平均时间: {df['Average Time'].mean():.1f}s")
            
            print(f"\n🏆 最佳迁移组合:")
            best_asr = df.loc[df['ASR'].idxmax()]
            print(f"  {best_asr['source_model']} → {best_asr['target_model']}: ASR={best_asr['ASR']:.3f}")
            
            worst_asr = df.loc[df['ASR'].idxmin()]
            print(f"  最差迁移组合: {worst_asr['source_model']} → {worst_asr['target_model']}: ASR={worst_asr['ASR']:.3f}")
            
            print(f"\n📊 按目标模型分析:")
            target_summary = df.groupby('target_model').agg({
                'ASR': ['mean', 'std'],
                'Average Queries': 'mean',
                'Average Time': 'mean'
            }).round(3)
            print(target_summary)

def interactive_mode():
    """交互式模式"""
    tester = TransferTester()
    
    print("🎯 迁移攻击测试系统")
    print("=" * 50)
    
    while True:
        print("\n选项:")
        print("1. 运行单个测试")
        print("2. 批量运行预设测试")
        print("3. 自定义批量测试")
        print("4. 查看历史结果")
        print("5. 生成可视化")
        print("6. 显示总结报告")
        print("7. 退出")
        
        choice = input("\n请选择 [1-7]: ").strip()
        
        if choice == "1":
            print("\n📋 可用模型:")
            print("源模型 (A):", [k for k, v in MODELS.items() if v['type'] in ['clm', 'mlm']])
            print("目标模型 (B):", [k for k, v in MODELS.items() if v['type'] == 'target'])
            
            source = input("源模型 (A): ").strip()
            target = input("目标模型 (B): ").strip()
            objective = input("优化目标 [ASR/ASR+GPT]: ").strip() or "ASR"
            
            if source in MODELS and target in MODELS:
                tester.run_single_transfer(source, target, objective)
            else:
                print("❌ 模型不存在")
        
        elif choice == "2":
            print("\n📋 预设测试:")
            print("1. 所有MLM→目标")
            print("2. 所有CLM→目标")
            print("3. 小规模测试")
            
            preset = input("选择 [1-3]: ").strip()
            objective = input("优化目标 [ASR/ASR+GPT]: ").strip() or "ASR"
            
            source_models = [k for k, v in MODELS.items() if v['type'] == ('mlm' if preset == '1' else 'clm')]
            target_models = [k for k, v in MODELS.items() if v['type'] == 'target']
            
            if preset == '3':
                # 小规模测试
                pairs = [('bert-large', 'mistral-7b'), ('llama2-7b', 'vicuna-7b')]
            else:
                pairs = [(s, t) for s in source_models for t in target_models]
            
            print(f"将运行 {len(pairs)} 个测试...")
            tester.run_batch_transfers(pairs, objective)
        
        elif choice == "3":
            print("\n📝 自定义测试 (格式: source,target)")
            print("输入空行结束")
            
            pairs = []
            while True:
                line = input("模型对: ").strip()
                if not line:
                    break
                try:
                    source, target = line.split(',')
                    if source in MODELS and target in MODELS:
                        pairs.append((source.strip(), target.strip()))
                    else:
                        print("❌ 模型不存在")
                except ValueError:
                    print("❌ 格式错误")
            
            if pairs:
                objective = input("优化目标 [ASR/ASR+GPT]: ").strip() or "ASR"
                tester.run_batch_transfers(pairs, objective)
        
        elif choice == "4":
            print("\n📋 模型列表:")
            for name, config in MODELS.items():
                print(f"  {name}: {config['description']} ({config['type']})")
        
        elif choice == "5":
            tester.create_visualizations()
        elif choice == "6":
            tester.show_summary_report()
        elif choice == "7":
            if tester.results:
                tester.save_results()
            print("👋 再见!")
            break

def main():
    parser = argparse.ArgumentParser(description="迁移攻击测试系统")
    parser.add_argument("--source", help="源模型 (A)")
    parser.add_argument("--target", help="目标模型 (B)")  
    parser.add_argument("--objective", default="ASR", help="优化目标")
    parser.add_argument("--batch", help="批量测试文件")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--output", default="./transfer_results", help="输出目录")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.source and args.target:
        tester = TransferTester(args.output)
        tester.run_single_transfer(args.source, args.target, args.objective)
        tester.create_heatmap()
    elif args.batch:
        tester = TransferTester(args.output)
        pairs = []
        with open(args.batch) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    s, t = line.strip().split(',')
                    pairs.append((s.strip(), t.strip()))
        tester.run_batch_transfers(pairs, args.objective)
        tester.create_heatmap()
    else:
        print("使用方式:")
        print("  python transfer_test.py --interactive    # 交互式")
        print("  python transfer_test.py --source A --target B")
        print("  python transfer_test.py --batch file.txt")

if __name__ == "__main__":
    main()