#!/usr/bin/env python3
"""
分析IF层参数设计
"""

import torch
from Models import modelpool
from Models.layer import IF

def main():
    print("🔍 分析IF层参数设计原理")
    print("="*60)
    
    # 1. 分析VGG16 classifier结构
    print("1. VGG16 Classifier 结构分析:")
    print("-" * 40)
    
    model = modelpool('vgg16', 'cifar10')
    
    print("Classifier Sequential:")
    for i, layer in enumerate(model.classifier):
        print(f"  [{i}] {layer}")
        if hasattr(layer, 'named_parameters'):
            for name, param in layer.named_parameters():
                print(f"      参数 {name}: {param.shape} = {param.numel()} 个参数")
    
    print("\n2. IF层设计原理解析:")
    print("-" * 40)
    
    # 查看IF层的参数设计
    if_layer = IF()
    print(f"IF层的thresh参数:")
    print(f"  形状: {if_layer.thresh.shape}")
    print(f"  数值: {if_layer.thresh.data}")
    print(f"  参数数量: {if_layer.thresh.numel()}")
    print(f"  是否可训练: {if_layer.thresh.requires_grad}")
    
    print("\n3. 为什么IF层参数只有1个？")
    print("-" * 40)
    print("🔑 关键原因:")
    print("  IF层的thresh是一个全局阈值参数，不是逐神经元的参数")
    print("  在QCFS方法中，每个IF层只需要一个共享的阈值")
    print("  这个阈值控制整个层的激活/量化行为")
    
    print("\n4. 对比不同层的Linear参数:")
    print("-" * 40)
    
    # 分析Linear层的参数
    linear1 = None  # 512 -> 4096
    linear2 = None  # 4096 -> 4096
    
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, torch.nn.Linear):
            if linear1 is None:
                linear1 = layer
                print(f"第1个Linear层 (索引{i}): {layer.weight.shape}")
                print(f"  权重参数: {layer.weight.numel():,} 个")
                print(f"  偏置参数: {layer.bias.numel() if layer.bias is not None else 0} 个")
            elif linear2 is None:
                linear2 = layer
                print(f"第2个Linear层 (索引{i}): {layer.weight.shape}")
                print(f"  权重参数: {layer.weight.numel():,} 个")
                print(f"  偏置参数: {layer.bias.numel() if layer.bias is not None else 0} 个")
                break
    
    print("\n5. IF层与Linear层的关系:")
    print("-" * 40)
    print("🧠 设计理念:")
    print("  • Linear层: 逐神经元的权重和偏置 (每个输出神经元有独立参数)")
    print("  • IF层: 全局阈值控制 (整个层共享一个阈值参数)")
    print("  • 这样设计的优势:")
    print("    - 参数数量少，减少过拟合风险")
    print("    - 易于硬件实现")
    print("    - 符合SNN的生物学原理")
    
    print("\n6. QCFS中IF层的作用:")
    print("-" * 40)
    print("在ANN模式 (T=0):")
    print("  x = x / thresh")
    print("  x = clamp(x, 0, 1)")  
    print("  x = floor(x*L+0.5)/L")
    print("  x = x * thresh")
    print("  → 实现分段线性量化")
    
    print("\n在SNN模式 (T>0):")
    print("  使用thresh作为脉冲发放阈值")
    print("  控制神经元的脉冲发放时机")
    print("  → 实现时间编码")
    
    print("\n7. 为什么不需要逐神经元阈值？")
    print("-" * 40)
    print("🎯 技术原因:")
    print("  1. QCFS方法追求ANN-SNN等价性")
    print("  2. 共享阈值减少了搜索空间")
    print("  3. 硬件实现更简单高效")
    print("  4. 避免了阈值参数的过度自由度")
    
    print("\n✅ 总结:")
    print("="*60)
    print("IF层的thresh参数设计为标量(1个参数)是QCFS方法的核心设计:")
    print("• 不是bug，而是特性!")
    print("• 每层一个阈值足以实现有效的量化和脉冲控制")
    print("• 这种设计在保持性能的同时大大减少了参数复杂度")


if __name__ == "__main__":
    main() 