#!/usr/bin/env python3
"""
集成改进Hessian重要性分析的训练测试
"""

import argparse
import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from improved_hessian_importance import ImprovedHessianWeightImportance


def main():
    """主函数"""
    print("="*80)
    print("🚀 测试改进的Hessian重要性计算器")
    print("="*80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # ANN模式
    model.to(device)
    
    # 加载数据
    train_loader, _ = datapool('cifar10', 16)
    
    # 创建改进的计算器
    hessian_calc = ImprovedHessianWeightImportance(
        model=model,
        device=device,
        n_samples=200,  # 增加采样数
        use_double_precision=True  # 使用双精度
    )
    
    # 运行分析
    criterion = nn.CrossEntropyLoss()
    results = hessian_calc.run_full_analysis(train_loader, criterion)
    
    # 获取剪枝候选
    pruning_candidates = results['pruning_candidates']
    
    # 按层分组统计
    layer_counts = {}
    for neuron in pruning_candidates:
        layer = neuron['layer']
        if layer not in layer_counts:
            layer_counts[layer] = 0
        layer_counts[layer] += 1
    
    print("\n建议剪枝的神经元统计:")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count} 个神经元")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # 可以保存结果供后续使用
    torch.save(results, 'hessian_importance_results.pt')
    print("📁 结果已保存到 hessian_importance_results.pt") 