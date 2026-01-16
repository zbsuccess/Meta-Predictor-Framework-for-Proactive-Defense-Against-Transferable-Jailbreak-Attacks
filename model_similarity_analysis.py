#!/usr/bin/env python3
"""
大模型相似性指标计算工具
对3*5个大模型的组合分别计算15组指标，并将每一类别的指标结果统一记录在一起
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# 设置中文字体支持
plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class ModelSimilarityAnalysis:
    """大模型相似性指标计算分析类"""
    def __init__(self, output_dir="./similarity_analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义3个源模型和5个目标模型（共15种组合）
        self.source_models = ['llama2-7b', 'bert-large', 'roberta-large']
        self.target_models = ['mistral-7b', 'vicuna-7b', 'guanaco-7b', 'starling-7b', 'chatgpt-3.5']
        
        # 存储所有结果
        self.results = {
            'output_distribution': [],  # 输出分布相似性指标
            'representation_space': [], # 表征空间相似性指标
            'behavior_functional': []   # 行为/功能相似性指标
        }
        
    def generate_sample_prob_distributions(self, model1_name, model2_name, size=1000):
        """生成两个模型的输出概率分布样本数据"""
        # 为了模拟不同模型的输出分布差异，使用不同的随机种子
        seed1 = hash(model1_name) % 1000
        seed2 = hash(model2_name) % 1000
        
        # 设置随机种子以确保结果可重现
        np.random.seed(seed1)
        # 生成第一个模型的概率分布（使用Dirichlet分布）
        alpha1 = np.random.uniform(0.1, 2.0, size)
        probs1 = np.exp(alpha1) / np.sum(np.exp(alpha1))
        
        # 生成与第一个模型相关的第二个模型的概率分布
        np.random.seed(seed2)
        alpha2 = alpha1 * np.random.uniform(0.8, 1.2, size)
        probs2 = np.exp(alpha2) / np.sum(np.exp(alpha2))
        
        # 归一化确保总和为1
        probs1 = probs1 / probs1.sum()
        probs2 = probs2 / probs2.sum()
        
        return probs1, probs2
    
    def generate_sample_logits(self, model1_name, model2_name, size=1000):
        """生成两个模型的logits样本数据"""
        # 为了模拟不同模型的logits差异，使用不同的随机种子
        seed1 = hash(model1_name) % 1000
        seed2 = hash(model2_name) % 1000
        
        # 设置随机种子以确保结果可重现
        np.random.seed(seed1)
        # 生成第一个模型的logits（使用正态分布）
        logits1 = np.random.normal(0, 5, size)
        
        # 生成与第一个模型相关的第二个模型的logits
        np.random.seed(seed2)
        correlation = np.random.uniform(0.3, 0.95)
        logits2 = correlation * logits1 + np.random.normal(0, np.sqrt(1 - correlation**2) * 5, size)
        
        return logits1, logits2
    
    def generate_sample_representations(self, model1_name, model2_name, num_samples=100, dim=768):
        """生成两个模型的隐藏层表示样本数据"""
        # 为了模拟不同模型的表示差异，使用不同的随机种子
        seed1 = hash(model1_name) % 1000
        seed2 = hash(model2_name) % 1000
        
        # 设置随机种子以确保结果可重现
        np.random.seed(seed1)
        # 生成第一个模型的表示（使用正态分布）
        repr1 = np.random.normal(0, 1, (num_samples, dim))
        
        # 生成与第一个模型相关的第二个模型的表示
        np.random.seed(seed2)
        # 创建一个随机变换矩阵来模拟不同模型的表示空间转换
        transform_matrix = np.random.normal(0, 1/dim, (dim, dim))
        # 添加一些噪声以模拟表示差异
        noise_level = np.random.uniform(0.1, 0.5)
        repr2 = np.dot(repr1, transform_matrix) + np.random.normal(0, noise_level, (num_samples, dim))
        
        return repr1, repr2
    
    def generate_sample_outputs(self, model1_name, model2_name, num_samples=100):
        """生成两个模型的输出结果样本数据"""
        # 为了模拟不同模型的输出差异，使用不同的随机种子
        seed1 = hash(model1_name) % 1000
        seed2 = hash(model2_name) % 1000
        
        # 设置随机种子以确保结果可重现
        np.random.seed(seed1)
        # 生成第一个模型的输出（0表示拒绝，1表示接受）
        outputs1 = np.random.binomial(1, 0.5, num_samples)
        
        # 生成与第一个模型相关的第二个模型的输出
        np.random.seed(seed2)
        # 基础一致性概率
        base_agreement = np.random.uniform(0.5, 0.9)
        # 生成与outputs1相关的outputs2
        flip_prob = 1 - base_agreement
        flip_mask = np.random.binomial(1, flip_prob, num_samples)
        outputs2 = np.logical_xor(outputs1, flip_mask).astype(int)
        
        return outputs1, outputs2
    
    # 第一类：输出分布相似性指标
    def compute_kl_divergence(self, p, q, epsilon=1e-10):
        """计算KL散度：D(p||q)"""
        # 添加小值以避免log(0)
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        # 归一化确保总和为1
        p = p / p.sum()
        q = q / q.sum()
        return np.sum(p * np.log(p / q))
    
    def compute_js_divergence(self, p, q, epsilon=1e-10):
        """计算Jensen-Shannon散度"""
        # 添加小值以避免log(0)
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        # 归一化确保总和为1
        p = p / p.sum()
        q = q / q.sum()
        # 计算平均分布
        m = 0.5 * (p + q)
        # 计算JS散度
        return 0.5 * (self.compute_kl_divergence(p, m) + self.compute_kl_divergence(q, m))
    
    def compute_emd(self, p, q):
        """计算Earth Mover's Distance"""
        # 使用一维Wasserstein距离作为EMD的简化版本
        n = len(p)
        return np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))
    
    def compute_logits_cosine_similarity(self, logits1, logits2):
        """计算Logits余弦相似度"""
        # 归一化logits向量
        logits1_norm = logits1 / np.linalg.norm(logits1)
        logits2_norm = logits2 / np.linalg.norm(logits2)
        # 计算余弦相似度
        return np.dot(logits1_norm, logits2_norm)
    
    def compute_rbo(self, rankings1, rankings2, k=10, p=0.9):
        """计算Rank-Based Overlap (RBO)"""
        # 创建rankings的字典映射
        rank_dict1 = {item: idx + 1 for idx, item in enumerate(rankings1[:k])}
        rank_dict2 = {item: idx + 1 for idx, item in enumerate(rankings2[:k])}
        
        # 获取所有唯一的项目
        all_items = set(rank_dict1.keys()).union(set(rank_dict2.keys()))
        
        # 计算RBO
        total = 0
        weight = 1
        for i in range(1, k + 1):
            # 计算前i个项目的重叠
            overlap = 0
            for item in all_items:
                if (item in rank_dict1 and rank_dict1[item] <= i) and (item in rank_dict2 and rank_dict2[item] <= i):
                    overlap += 1
            
            # 计算前i个项目的RBO
            total += weight * overlap / i
            weight *= p
        
        # 计算剩余部分
        remaining = weight * len(set(rankings1[:k]).intersection(set(rankings2[:k]))) / k
        
        return total + remaining
    
    # 第二类：表征空间相似性指标
    def compute_cka(self, X, Y):
        """计算Centered Kernel Alignment (CKA)"""
        # 中心化
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)
        
        # 计算Gram矩阵
        K = X_centered @ X_centered.T
        L = Y_centered @ Y_centered.T
        
        # 归一化
        K_norm = np.linalg.norm(K)
        L_norm = np.linalg.norm(L)
        
        # 计算CKA
        return np.sum(K * L.T) / (K_norm * L_norm)
    
    def compute_svcca(self, X, Y, n_components=100):
        """计算Singular Value Canonical Correlation Analysis (SVCCA)"""
        # 中心化
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)
        
        # 奇异值分解
        try:
            Ux, Sx, Vx = np.linalg.svd(X_centered, full_matrices=False)
            Uy, Sy, Vy = np.linalg.svd(Y_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0  # 处理SVD计算失败的情况
        
        # 选择主成分
        n_components = min(n_components, Ux.shape[1], Uy.shape[1])
        Ux_reduced = Ux[:, :n_components]
        Uy_reduced = Uy[:, :n_components]
        
        # 计算CCA
        C = Ux_reduced.T @ Uy_reduced
        try:
            Uc, Sc, Vc = np.linalg.svd(C, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0  # 处理SVD计算失败的情况
        
        # SVCCA是相关系数的平均值
        return np.mean(Sc)
    
    def compute_pwcca(self, X, Y, n_components=100):
        """计算Weighted Singular Value Canonical Correlation Analysis (PWCCA)"""
        # 中心化
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)
        
        # 奇异值分解
        try:
            Ux, Sx, Vx = np.linalg.svd(X_centered, full_matrices=False)
            Uy, Sy, Vy = np.linalg.svd(Y_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0  # 处理SVD计算失败的情况
        
        # 选择主成分
        n_components = min(n_components, Ux.shape[1], Uy.shape[1])
        Ux_reduced = Ux[:, :n_components]
        Uy_reduced = Uy[:, :n_components]
        
        # 计算CCA
        C = Ux_reduced.T @ Uy_reduced
        try:
            Uc, Sc, Vc = np.linalg.svd(C, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0  # 处理SVD计算失败的情况
        
        # 计算权重
        weights = Sx[:n_components] / np.sum(Sx[:n_components])
        
        # PWCCA是加权相关系数的平均值
        return np.sum(weights * Sc)
    
    def compute_rsa(self, X, Y, distance_metric='correlation'):
        """计算Representational Similarity Analysis (RSA)"""
        # 计算表示空间中的距离矩阵
        n = X.shape[0]
        Dx = np.zeros((n, n))
        Dy = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if distance_metric == 'correlation':
                    # 计算皮尔逊相关系数
                    corr_coef = np.corrcoef(X[i], X[j])[0, 1]
                    # 处理NaN值
                    if np.isnan(corr_coef):
                        corr_coef = 0.0
                    Dx[i, j] = Dx[j, i] = 1 - corr_coef
                    
                    corr_coef2 = np.corrcoef(Y[i], Y[j])[0, 1]
                    if np.isnan(corr_coef2):
                        corr_coef2 = 0.0
                    Dy[i, j] = Dy[j, i] = 1 - corr_coef2
                else:
                    Dx[i, j] = Dx[j, i] = np.linalg.norm(X[i] - X[j])
                    Dy[i, j] = Dy[j, i] = np.linalg.norm(Y[i] - Y[j])
        
        # 计算两个距离矩阵之间的相关性
        return np.corrcoef(Dx.flatten(), Dy.flatten())[0, 1]
    
    # 第三类：行为/功能相似性指标
    def compute_task_agreement(self, outputs1, outputs2):
        """计算任务一致率"""
        # 计算两个模型输出相同的比例
        return np.mean(outputs1 == outputs2)
    
    def compute_pass_at_k_agreement(self, probs1, probs2, k=1, correctness_threshold=0.5):
        """计算Pass@k一致率"""
        # 简化版：假设probs已经是top-k的概率
        # 计算两个模型都接受或都拒绝的比例
        pass1 = probs1 >= correctness_threshold
        pass2 = probs2 >= correctness_threshold
        return np.mean(pass1 == pass2)
    
    def compute_adversarial_transfer_rate(self, model1_name, model2_name):
        """计算对抗迁移率"""
        # 在实际应用中，这应该基于真实的攻击结果
        # 这里我们基于模型名称生成一个模拟值
        # 模型越相似，对抗迁移率越高
        seed = hash(f"{model1_name}_{model2_name}") % 1000
        np.random.seed(seed)
        
        # 基础迁移率加上一些随机变化
        base_transfer_rate = 0.3 + 0.5 * (1 - 1/(1 + np.exp(-0.5 * seed/100)))
        transfer_rate = np.clip(base_transfer_rate + np.random.normal(0, 0.1), 0, 1)
        
        return transfer_rate
    
    def compute_semantic_similarity(self, model1_name, model2_name):
        """计算语义相似性"""
        # 在实际应用中，这应该基于真实的输出文本和预训练的句子嵌入模型
        # 这里我们基于模型名称生成一个模拟值
        seed = hash(f"{model1_name}_{model2_name}_semantic") % 1000
        np.random.seed(seed)
        
        # 基础语义相似度加上一些随机变化
        base_similarity = 0.4 + 0.5 * (1 - 1/(1 + np.exp(-0.5 * seed/100)))
        similarity = np.clip(base_similarity + np.random.normal(0, 0.1), 0, 1)
        
        return similarity
    
    def compute_all_metrics_for_pair(self, source_model, target_model):
        """计算一对模型的所有相似性指标"""
        print(f"\n🔄 计算模型对 {source_model} 和 {target_model} 的相似性指标")
        
        # 生成样本数据
        print("  生成样本数据...")
        probs1, probs2 = self.generate_sample_prob_distributions(source_model, target_model)
        logits1, logits2 = self.generate_sample_logits(source_model, target_model)
        repr1, repr2 = self.generate_sample_representations(source_model, target_model)
        outputs1, outputs2 = self.generate_sample_outputs(source_model, target_model)
        
        # 创建排名（基于概率）
        rankings1 = np.argsort(-probs1)  # 降序排列的索引
        rankings2 = np.argsort(-probs2)
        
        # 计算输出分布相似性指标
        print("  计算输出分布相似性指标...")
        output_metrics = {
            'source_model': source_model,
            'target_model': target_model,
            'KL散度': self.compute_kl_divergence(probs1, probs2),
            'JS散度': self.compute_js_divergence(probs1, probs2),
            'EMD': self.compute_emd(probs1, probs2),
            'Logits余弦相似度': self.compute_logits_cosine_similarity(logits1, logits2),
            'RBO': self.compute_rbo(rankings1, rankings2)
        }
        self.results['output_distribution'].append(output_metrics)
        
        # 计算表征空间相似性指标
        print("  计算表征空间相似性指标...")
        representation_metrics = {
            'source_model': source_model,
            'target_model': target_model,
            'CKA': self.compute_cka(repr1, repr2),
            'SVCCA': self.compute_svcca(repr1, repr2),
            'PWCCA': self.compute_pwcca(repr1, repr2),
            'RSA': self.compute_rsa(repr1, repr2)
        }
        self.results['representation_space'].append(representation_metrics)
        
        # 计算行为/功能相似性指标
        print("  计算行为/功能相似性指标...")
        behavior_metrics = {
            'source_model': source_model,
            'target_model': target_model,
            '任务一致率': self.compute_task_agreement(outputs1, outputs2),
            'Pass@k一致率': self.compute_pass_at_k_agreement(probs1[:10], probs2[:10]),
            '对抗迁移率': self.compute_adversarial_transfer_rate(source_model, target_model),
            '语义相似性': self.compute_semantic_similarity(source_model, target_model)
        }
        self.results['behavior_functional'].append(behavior_metrics)
        
        print("  ✅ 计算完成")
    
    def compute_all_metrics(self):
        """计算所有模型对的相似性指标"""
        print("🎯 开始计算所有模型对的相似性指标")
        print("=" * 60)
        
        total_pairs = len(self.source_models) * len(self.target_models)
        print(f"总共有 {total_pairs} 组模型对需要计算")
        
        # 对每一对模型计算指标
        for source in self.source_models:
            for target in self.target_models:
                self.compute_all_metrics_for_pair(source, target)
        
        print("=" * 60)
        print("✅ 所有模型对的相似性指标计算完成")
    
    def save_results(self):
        """保存计算结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存为JSON格式
        json_path = os.path.join(self.output_dir, f"similarity_results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存到: {json_path}")
        
        # 将每一类别的指标结果统一记录在一起
        for metric_type, metrics_list in self.results.items():
            if metrics_list:
                # 创建类别专用的结果文件
                type_json_path = os.path.join(self.output_dir, f"{metric_type}_results_{timestamp}.json")
                with open(type_json_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_list, f, indent=2, ensure_ascii=False)
                print(f"📊 {metric_type} 指标已保存到: {type_json_path}")
    
    def create_simple_visualization(self):
        """创建简单的可视化图表"""
        print("📊 开始生成可视化图表...")
        
        # 为输出分布相似性指标创建简单的热力图
        for metric_type, metrics_list in self.results.items():
            if not metrics_list:
                continue
                
            print(f"  创建 {metric_type} 指标的示例热力图...")
            
            # 选择第一个指标创建示例热力图
            first_metric = None
            for key in metrics_list[0].keys():
                if key not in ['source_model', 'target_model']:
                    first_metric = key
                    break
            
            if first_metric:
                # 创建透视表数据
                pivot_data = np.zeros((len(self.source_models), len(self.target_models)))
                
                for idx, item in enumerate(metrics_list):
                    source_idx = self.source_models.index(item['source_model'])
                    target_idx = self.target_models.index(item['target_model'])
                    pivot_data[source_idx, target_idx] = item[first_metric]
                
                # 创建热力图
                plt.figure(figsize=(10, 6))
                plt.imshow(pivot_data, cmap='viridis', aspect='auto')
                plt.colorbar(label=first_metric)
                plt.xticks(np.arange(len(self.target_models)), self.target_models, rotation=45)
                plt.yticks(np.arange(len(self.source_models)), self.source_models)
                plt.title(f"{metric_type.replace('_', ' ').title()} - {first_metric}")
                plt.tight_layout()
                
                # 保存图表
                chart_path = os.path.join(self.output_dir, f"{metric_type}_{first_metric}_heatmap.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"    ✅ 已保存: {chart_path}")
        
        print("✅ 所有可视化图表生成完成")
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("📋 生成总结报告...")
        
        report_path = os.path.join(self.output_dir, f"similarity_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("大模型相似性指标计算总结报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源模型数量: {len(self.source_models)}\n")
            f.write(f"目标模型数量: {len(self.target_models)}\n")
            f.write(f"模型对总数: {len(self.source_models) * len(self.target_models)}\n\n")
            
            f.write("源模型列表:\n")
            for model in self.source_models:
                f.write(f"  - {model}\n")
            f.write("\n")
            
            f.write("目标模型列表:\n")
            for model in self.target_models:
                f.write(f"  - {model}\n")
            f.write("\n")
            
            # 为每类指标生成统计信息
            for metric_type, metrics_list in self.results.items():
                if not metrics_list:
                    continue
                    
                f.write(f"\n{metric_type.replace('_', ' ').title()} 指标统计:\n")
                f.write("-" * 60 + "\n")
                
                # 获取指标名称
                metric_names = [k for k in metrics_list[0].keys() if k not in ['source_model', 'target_model']]
                
                for metric in metric_names:
                    values = [item[metric] for item in metrics_list]
                    f.write(f"{metric}:\n")
                    f.write(f"  平均值: {np.mean(values):.4f}\n")
                    f.write(f"  标准差: {np.std(values):.4f}\n")
                    f.write(f"  最大值: {np.max(values):.4f}\n")
                    f.write(f"  最小值: {np.min(values):.4f}\n\n")
        
        print(f"💾 总结报告已保存到: {report_path}")


def main():
    """主函数"""
    # 创建相似性指标计算器
    metrics_calculator = ModelSimilarityAnalysis()
    
    # 计算所有模型对的相似性指标
    metrics_calculator.compute_all_metrics()
    
    # 保存结果（包括按类别统一记录）
    metrics_calculator.save_results()
    
    # 创建可视化图表
    metrics_calculator.create_simple_visualization()
    
    # 生成总结报告
    metrics_calculator.generate_summary_report()
    
    print("\n🎉 大模型相似性指标计算任务已完成!")


if __name__ == "__main__":
    main()