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
    print("🧪 集成改进Hessian重要性分析的训练测试")
    print("="*80)
    
    # 参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建VGG16模型...")
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # ANN模式分析
    model.to(device)
    
    # 加载数据
    print("加载CIFAR-10数据...")
    train_loader, test_loader = datapool('cifar10', 32)
    
    # 模拟一些训练步骤
    print("\n模拟训练过程...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 5:  # 只训练几个batch
            break
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}")
    
    # 在训练后分析IF层重要性
    print("\n🔍 开始Hessian重要性分析...")
    hessian_calc = ImprovedHessianWeightImportance(
        model=model,
        device=device,
        n_samples=100,  # 适中的采样数，平衡精度和速度
        use_double_precision=True
    )
    
    # 运行分析
    model.eval()
    results = hessian_calc.run_full_analysis(train_loader, criterion)
    
    # 应用剪枝建议
    print("\n🎯 应用剪枝建议:")
    print("="*50)
    
    pruning_candidates = results['pruning_candidates']
    print(f"建议剪枝的IF层 (共{len(pruning_candidates)}个):")
    
    for layer_name, importance in pruning_candidates:
        print(f"  ✂️  {layer_name}: 重要性 = {importance:.8f}")
    
    # 可以在这里实际应用剪枝
    # 例如，设置某些IF层的阈值或禁用它们
    
    print("\n📈 重要性统计:")
    valid_importances = results['valid_importances']
    if len(valid_importances) > 0:
        print(f"  有效IF层数: {len(valid_importances)}")
        print(f"  平均重要性: {sum(valid_importances)/len(valid_importances):.8f}")
        print(f"  重要性范围: [{min(valid_importances):.8f}, {max(valid_importances):.8f}]")
        
        # 按重要性排序显示
        layer_importance_pairs = [(name, imp[0]) for name, imp in results['weight_importance'].items()]
        layer_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n  🏆 重要性排行榜 (Top 5):")
        for i, (name, importance) in enumerate(layer_importance_pairs[:5]):
            print(f"    {i+1}. {name}: {importance:.8f}")
        
        print("\n  ⚠️  最不重要 (Bottom 3):")
        for i, (name, importance) in enumerate(layer_importance_pairs[-3:]):
            print(f"    {len(layer_importance_pairs)-2+i}. {name}: {importance:.8f}")
    
    print("\n✅ 测试完成！")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
    
    # 可以保存结果供后续使用
    torch.save(results, 'hessian_importance_results.pt')
    print("📁 结果已保存到 hessian_importance_results.pt") 