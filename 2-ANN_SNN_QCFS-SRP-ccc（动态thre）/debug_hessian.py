#!/usr/bin/env python3
"""
调试Hessian权重重要性计算 - 分析为什么重要性值全为0
"""

import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from hessian_importance import HessianWeightImportance
import numpy as np


def debug_simple_case():
    """使用简单模型调试"""
    print("="*60)
    print("调试简单模型")
    print("="*60)
    
    # 创建一个简单的2层网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 创建简单数据
    batch_size = 4
    data = torch.randn(batch_size, 10).to(device)
    target = torch.randint(0, 2, (batch_size,)).to(device)
    
    # 前向传播计算梯度
    model.zero_grad()
    output = model(data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward(create_graph=True)
    
    print("模型参数和梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: 参数形状={param.shape}, 梯度范数={param.grad.norm():.6f}")
            print(f"  参数范数={param.norm():.6f}")
            print(f"  权重前几个值: {param.data.flatten()[:5]}")
            print(f"  梯度前几个值: {param.grad.flatten()[:5]}")
        else:
            print(f"{name}: 无梯度")
    
    # 手动计算一个简单的Hessian-向量乘积
    print("\n手动计算Hessian-向量乘积:")
    params = [param for param in model.parameters() if param.grad is not None]
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    
    # 生成随机向量
    v = [torch.randn_like(p) for p in params]
    
    print("随机向量:")
    for i, vi in enumerate(v):
        print(f"  v[{i}]: 形状={vi.shape}, 范数={vi.norm():.6f}")
    
    # 计算Hessian-向量乘积
    try:
        hv = torch.autograd.grad(grads, params, grad_outputs=v, 
                               only_inputs=True, retain_graph=False)
        
        print("Hessian-向量乘积结果:")
        for i, hvi in enumerate(hv):
            print(f"  Hv[{i}]: 形状={hvi.shape}, 范数={hvi.norm():.6f}")
            
        # 计算 v^T * H * v
        trace_estimate = sum([torch.sum(vi * hvi) for vi, hvi in zip(v, hv)]).item()
        print(f"trace估计: {trace_estimate:.6f}")
        
        # 计算权重重要性（简化版）
        for i, (param, hvi, vi) in enumerate(zip(params, hv, v)):
            # 按第一个神经元计算
            if len(param.shape) >= 2:
                channel_weight = param[0]  # 第一个输出神经元
                weight_norm_sq = channel_weight.norm(p=2) ** 2
                num_weights = channel_weight.numel()
                
                # 计算这个通道的trace
                channel_hv = hvi[0].flatten()
                channel_v = vi[0].flatten()
                channel_trace = channel_hv.dot(channel_v).item()
                
                importance = channel_trace * (weight_norm_sq.item() / num_weights)
                
                print(f"参数{i} 第0个通道:")
                print(f"  channel_trace: {channel_trace:.6f}")
                print(f"  weight_norm_sq: {weight_norm_sq:.6f}")
                print(f"  num_weights: {num_weights}")
                print(f"  importance: {importance:.6f}")
        
    except Exception as e:
        print(f"计算Hessian-向量乘积失败: {e}")


def debug_vgg_case():
    """调试VGG模型的一个小batch"""
    print("\n" + "="*60)
    print("调试VGG模型")
    print("="*60)
    
    # 创建VGG模型
    model = modelpool('vgg16', 'cifar10')
    model.set_T(0)  # 使用ANN模式，避免SNN复杂性
    model.set_L(8)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # 创建小batch数据
    batch_size = 4
    data = torch.randn(batch_size, 3, 32, 32).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)
    
    # 前向传播
    model.zero_grad()
    output = model(data)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward(create_graph=True)
    
    print(f"损失值: {loss.item():.6f}")
    
    # 检查前几层的梯度
    layer_count = 0
    for name, param in model.named_parameters():
        if 'weight' in name and layer_count < 3:  # 只看前3层
            if param.grad is not None:
                print(f"\n层 {name}:")
                print(f"  参数形状: {param.shape}")
                print(f"  参数范数: {param.norm():.6f}")
                print(f"  梯度范数: {param.grad.norm():.6f}")
                print(f"  梯度最大值: {param.grad.max():.6f}")
                print(f"  梯度最小值: {param.grad.min():.6f}")
                
                # 手动计算第一个通道的重要性
                if len(param.shape) >= 2:
                    channel_weight = param[0]  # 第一个输出通道
                    weight_norm_sq = channel_weight.norm(p=2) ** 2
                    num_weights = channel_weight.numel()
                    
                    print(f"  第0通道权重范数平方: {weight_norm_sq:.6f}")
                    print(f"  第0通道权重数量: {num_weights}")
                    print(f"  预期重要性系数: {weight_norm_sq.item() / num_weights:.6f}")
                
                layer_count += 1
            else:
                print(f"层 {name}: 无梯度")


def main():
    print("🔍 调试Hessian权重重要性计算")
    print("目标：找出为什么所有重要性值都是0")
    
    # 1. 调试简单模型
    debug_simple_case()
    
    # 2. 调试VGG模型
    debug_vgg_case()
    
    print("\n" + "="*60)
    print("🎯 可能的问题和解决方案:")
    print("1. Hessian trace估计值太小")
    print("2. 权重初始化导致的数值问题")
    print("3. SNN模式下的梯度计算问题")
    print("4. Hutchinson采样数量不足")
    print("5. 数值精度问题")
    print("="*60)


if __name__ == '__main__':
    main() 