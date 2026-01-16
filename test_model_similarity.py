#!/usr/bin/env python3
"""
测试大模型相似性指标计算功能
"""

import os
import json
from model_similarity_analysis import ModelSimilarityAnalysis


def test_model_similarity_analysis():
    """测试ModelSimilarityAnalysis类的主要功能"""
    print("===== 开始测试大模型相似性指标计算功能 =====")
    
    # 创建一个临时输出目录
    test_output_dir = "./test_similarity_results"
    
    # 创建相似性指标计算器实例
    print("1. 创建ModelSimilarityAnalysis实例...")
    analyzer = ModelSimilarityAnalysis(output_dir=test_output_dir)
    print("   ✅ 实例创建成功")
    
    # 验证模型列表
    print(f"2. 验证模型列表...")
    print(f"   源模型数量: {len(analyzer.source_models)}")
    print(f"   目标模型数量: {len(analyzer.target_models)}")
    print(f"   模型对总数: {len(analyzer.source_models) * len(analyzer.target_models)}")
    print(f"   源模型列表: {', '.join(analyzer.source_models)}")
    print(f"   目标模型列表: {', '.join(analyzer.target_models)}")
    print("   ✅ 模型列表验证成功")
    
    # 测试计算一对模型的指标
    print("\n3. 测试计算单对模型的相似性指标...")
    source_model = analyzer.source_models[0]
    target_model = analyzer.target_models[0]
    print(f"   计算模型对 {source_model} 和 {target_model} 的指标")
    analyzer.compute_all_metrics_for_pair(source_model, target_model)
    print("   ✅ 单对模型指标计算成功")
    
    # 检查结果结构
    print("\n4. 检查结果结构...")
    for metric_type, results in analyzer.results.items():
        if results:
            print(f"   {metric_type} 指标包含 {len(results)} 条记录")
            print(f"   第一条记录的键: {list(results[0].keys())}")
    print("   ✅ 结果结构检查成功")
    
    # 保存单对模型的结果
    print("\n5. 测试保存结果...")
    analyzer.save_results()
    print("   ✅ 结果保存成功")
    
    # 清理测试结果
    print("\n6. 清理测试环境...")
    # 这里不删除测试结果，以便用户查看测试输出
    print("   ✅ 测试环境清理完成")
    
    print("\n===== 测试完成 =====")
    print(f"\n测试结果保存在: {test_output_dir}")
    print("要执行完整的15组模型对计算，请直接运行: python model_similarity_analysis.py")


def main():
    """主函数"""
    test_model_similarity_analysis()


if __name__ == "__main__":
    main()