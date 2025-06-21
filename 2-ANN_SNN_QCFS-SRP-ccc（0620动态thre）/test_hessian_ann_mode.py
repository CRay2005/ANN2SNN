#!/usr/bin/env python3
"""
在ANN模式下测试Hessian权重重要性计算
避免SNN模式的复杂性
"""

import argparse
import os
import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from hessian_importance import HessianWeightImportance


def main():
    print("🧪 ANN模式下的Hessian权重重要性测试")
    print("="*60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = datapool('cifar10', 16)
    
    # 创建模型 - 关键：使用ANN模式
    print("创建模型（ANN模式）...")
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # 🔑 关键：设置T=0使用ANN模式
    model.to(device)
    
    print(f"模型时间步长: T={model.T}")
    
    # 创建Hessian权重重要性计算器
    print("创建Hessian权重重要性计算器...")
    hessian_calculator = HessianWeightImportance(
        model=model, 
        device=device,
        n_samples=20  # 使用较少的采样数快速测试
    )
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 先测试模型前向传播
    print("\n测试模型前向传播...")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 0:
            break
        
        data, target = data.to(device), target.to(device)
        
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        
        print(f"输出形状: {output.shape}")
        print(f"损失值: {loss.item():.6f}")
        
        # 检查前几层的梯度
        layer_count = 0
        for name, param in model.named_parameters():
            if 'weight' in name and layer_count < 3:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    print(f"层 {name}: 参数范数={param_norm:.6f}, 梯度范数={grad_norm:.6f}")
                    
                    # 检查是否有非零梯度
                    nonzero_grads = (param.grad != 0).sum().item()
                    total_params = param.grad.numel()
                    print(f"  非零梯度: {nonzero_grads}/{total_params} ({nonzero_grads/total_params*100:.2f}%)")
                    
                    layer_count += 1
        break
    
    # 运行完整分析
    print("\n开始Hessian权重重要性分析（ANN模式）...")
    results = hessian_calculator.run_full_analysis(
        data_loader=train_loader,
        criterion=criterion
    )
    
    # 分析结果
    print("\n📊 结果分析:")
    print("="*60)
    
    # 统计非零重要性
    total_channels = 0
    nonzero_channels = 0
    all_importances = []
    
    for name, importance_list in results['weight_importance'].items():
        total_channels += len(importance_list)
        nonzero_count = sum(1 for imp in importance_list if abs(imp) > 1e-10)
        nonzero_channels += nonzero_count
        all_importances.extend(importance_list)
        
        print(f"{name}: {nonzero_count}/{len(importance_list)} 非零重要性")
        if nonzero_count > 0:
            nonzero_importances = [imp for imp in importance_list if abs(imp) > 1e-10]
            print(f"  非零重要性范围: [{min(nonzero_importances):.8f}, {max(nonzero_importances):.8f}]")
    
    print(f"\n总体统计:")
    print(f"总通道数: {total_channels}")
    print(f"非零重要性通道: {nonzero_channels}")
    print(f"非零比例: {nonzero_channels/total_channels*100:.2f}%")
    
    if nonzero_channels > 0:
        nonzero_all = [imp for imp in all_importances if abs(imp) > 1e-10]
        print(f"非零重要性统计:")
        print(f"  均值: {sum(nonzero_all)/len(nonzero_all):.8f}")
        print(f"  范围: [{min(nonzero_all):.8f}, {max(nonzero_all):.8f}]")
        
        # 显示一些具体的非零重要性值
        print(f"前10个非零重要性值:")
        sorted_nonzero = sorted(nonzero_all, key=abs, reverse=True)
        for i, imp in enumerate(sorted_nonzero[:10]):
            print(f"  {i+1}. {imp:.8f}")
    else:
        print("❌ 所有重要性值仍然为0")
        print("可能需要检查：")
        print("1. 模型是否正确初始化")
        print("2. 梯度计算是否正确")
        print("3. Hutchinson采样实现")
        print("4. 数值精度问题")
    
    return results


if __name__ == '__main__':
    main() 