 #!/usr/bin/env python3
"""
测试基于HSTNN的Hessian权重重要性计算
"""

import argparse
import os
import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from hessian_importance import HessianWeightImportance


def main():
    parser = argparse.ArgumentParser(description='测试Hessian权重重要性计算')
    parser.add_argument('-data', '--dataset', default='cifar10', type=str, help='数据集')
    parser.add_argument('-arch', '--model', default='vgg16', type=str, help='模型架构')
    parser.add_argument('-T', '--time', default=4, type=int, help='SNN时间步长')
    parser.add_argument('-L', '--L', default=8, type=int, help='量化级别')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='批大小')
    parser.add_argument('--device', default='0', type=str, help='GPU设备')
    parser.add_argument('--n_samples', default=50, type=int, help='Hutchinson采样数')
    
    args = parser.parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    
    # 创建模型
    print("创建模型...")
    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)
    model.set_T(args.time)
    model.to(device)
    
    # 打印模型结构
    print("\n模型结构概览:")
    print("="*50)
    total_params = 0
    conv_params = 0
    fc_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_params = module.weight.numel()
            total_params += num_params
            
            if isinstance(module, nn.Conv2d):
                conv_params += num_params
                layer_type = "Conv2d"
            else:
                fc_params += num_params
                layer_type = "Linear"
            
            print(f"{name} ({layer_type}): {module.weight.shape} - {num_params:,} 参数")
    
    print(f"\n参数统计:")
    print(f"  卷积层参数: {conv_params:,}")
    print(f"  全连接层参数: {fc_params:,}")
    print(f"  总参数: {total_params:,}")
    print("="*50)
    
    # 创建Hessian权重重要性计算器
    print(f"\n创建Hessian权重重要性计算器 (采样数: {args.n_samples})...")
    hessian_calculator = HessianWeightImportance(
        model=model, 
        device=device,
        n_samples=args.n_samples
    )
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 运行完整分析
    print("\n开始Hessian权重重要性分析...")
    results = hessian_calculator.run_full_analysis(
        data_loader=train_loader,
        criterion=criterion
    )
    
    # 详细分析结果
    print("\n📊 详细结果分析:")
    print("="*80)
    
    # 分析层级重要性
    layer_importance_summary = {}
    for name, importance_list in results['weight_importance'].items():
        layer_importance_summary[name] = {
            'mean_importance': sum(importance_list) / len(importance_list),
            'num_channels': len(importance_list),
            'total_importance': sum(importance_list)
        }
    
    # 按平均重要性排序
    sorted_layers = sorted(layer_importance_summary.items(), 
                          key=lambda x: x[1]['mean_importance'], reverse=True)
    
    print("各层平均重要性排序:")
    print(f"{'层名':<30} {'平均重要性':<15} {'通道数':<10} {'总重要性':<15}")
    print("-" * 80)
    
    for name, stats in sorted_layers:
        print(f"{name:<30} {stats['mean_importance']:<15.6f} "
              f"{stats['num_channels']:<10} {stats['total_importance']:<15.6f}")
    
    # 分析卷积层vs全连接层
    conv_importances = []
    fc_importances = []
    
    for name, importance_list in results['weight_importance'].items():
        if 'layer' in name:  # 卷积层
            conv_importances.extend(importance_list)
        elif 'classifier' in name:  # 全连接层
            fc_importances.extend(importance_list)
    
    if conv_importances and fc_importances:
        print(f"\n📈 层类型对比:")
        print(f"卷积层重要性: 均值={sum(conv_importances)/len(conv_importances):.6f}, "
              f"通道数={len(conv_importances)}")
        print(f"全连接层重要性: 均值={sum(fc_importances)/len(fc_importances):.6f}, "
              f"神经元数={len(fc_importances)}")
    
    # 剪枝建议
    print(f"\n✂️ 剪枝建议:")
    pruning_candidates = results['pruning_candidates']
    print("建议优先剪枝的层（重要性最低）:")
    
    for name, channels in list(pruning_candidates.items())[:5]:  # 显示前5层
        avg_importance = sum([imp for _, imp in channels]) / len(channels)
        print(f"  {name}: {len(channels)} 个通道, 平均重要性={avg_importance:.6f}")
    
    print(f"\n🎯 核心发现:")
    print(f"1. 成功实现了 weight_importance = hessian_trace * (weight_norm^2 / num_weights)")
    print(f"2. 通过Hutchinson方法高效估计了Hessian trace")
    print(f"3. 实现了channel-wise的细粒度重要性分析")
    print(f"4. 为SNN剪枝提供了基于二阶信息的科学依据")
    
    return results


if __name__ == '__main__':
    main()