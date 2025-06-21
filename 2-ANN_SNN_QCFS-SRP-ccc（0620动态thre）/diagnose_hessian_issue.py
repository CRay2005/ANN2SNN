#!/usr/bin/env python3
"""
诊断Hessian权重重要性为0的根本原因
"""

import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from Models.layer import IF


def diagnose_if_layer_issue():
    """诊断IF层的问题"""
    print("="*80)
    print("🔍 诊断IF层Hessian重要性为0的问题")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # ANN模式
    model.to(device)
    
    # 加载数据
    train_loader, _ = datapool('cifar10', 16)
    
    # 获取一个batch
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 0:
            break
        
        data, target = data.to(device), target.to(device)
        
        # 1. 检查IF层参数初始化状态
        print("1. 检查IF层参数初始化状态:")
        print("-" * 40)
        
        if_params = []
        for name, module in model.named_modules():
            if isinstance(module, IF):
                for param_name, param in module.named_parameters():
                    full_name = f"{name}.{param_name}"
                    if_params.append((full_name, param))
                    print(f"IF层 {full_name}:")
                    print(f"  形状: {param.shape}")
                    print(f"  值: {param.data}")
                    print(f"  范数: {param.norm():.6f}")
                    print(f"  requires_grad: {param.requires_grad}")
        
        # 2. 测试前向传播和梯度计算
        print("\n2. 测试前向传播和梯度计算:")
        print("-" * 40)
        
        model.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward(create_graph=True)
        
        print(f"损失值: {loss.item():.6f}")
        print(f"输出形状: {output.shape}")
        
        # 检查IF层梯度
        print("\n3. 检查IF层梯度:")
        print("-" * 40)
        
        for name, param in if_params:
            if param.grad is not None:
                print(f"IF层 {name}:")
                print(f"  梯度值: {param.grad.data}")
                print(f"  梯度范数: {param.grad.norm():.6f}")
                print(f"  梯度是否为0: {torch.allclose(param.grad, torch.zeros_like(param.grad))}")
            else:
                print(f"IF层 {name}: 无梯度！")
        
        # 4. 测试简单的Hessian计算
        print("\n4. 测试简单的Hessian计算:")
        print("-" * 40)
        
        # 选择第一个IF层进行测试
        if if_params:
            test_name, test_param = if_params[0]
            print(f"测试IF层: {test_name}")
            
            if test_param.grad is not None:
                # 生成随机向量
                v = torch.randn_like(test_param)
                print(f"随机向量v: {v}")
                print(f"v的范数: {v.norm():.6f}")
                
                try:
                    # 计算Hessian-向量乘积
                    hv = torch.autograd.grad([test_param.grad], [test_param], 
                                           grad_outputs=[v], retain_graph=True)[0]
                    print(f"Hessian-向量乘积Hv: {hv}")
                    print(f"Hv的范数: {hv.norm():.6f}")
                    
                    # 计算trace
                    trace = torch.sum(v * hv).item()
                    print(f"trace估计 (v^T * H * v): {trace:.6f}")
                    
                    # 计算重要性
                    weight_norm_sq = test_param.norm(p=2) ** 2
                    importance = trace * weight_norm_sq.item()
                    print(f"权重范数平方: {weight_norm_sq:.6f}")
                    print(f"重要性 (trace * weight_norm^2): {importance:.6f}")
                    
                except Exception as e:
                    print(f"计算Hessian失败: {e}")
            else:
                print("没有梯度，无法计算Hessian")
        
        # 5. 分析问题原因
        print("\n5. 问题分析:")
        print("-" * 40)
        
        # 检查IF层是否处于激活状态
        if_layers = []
        for name, module in model.named_modules():
            if isinstance(module, IF):
                if_layers.append((name, module))
        
        print(f"发现 {len(if_layers)} 个IF层")
        
        # 测试IF层在ANN模式下的行为
        print("\n测试IF层在ANN模式下的行为:")
        test_input = torch.randn(4, 64, 32, 32).to(device)  # 模拟中间层输入
        
        for name, if_layer in if_layers[:3]:  # 只测试前3个
            print(f"\nIF层 {name}:")
            print(f"  T (时间步): {if_layer.T}")
            print(f"  thresh: {if_layer.thresh}")
            
            try:
                # 测试IF层的前向传播
                if_layer.eval()
                with torch.no_grad():
                    test_output = if_layer(test_input)
                    print(f"  输入形状: {test_input.shape}")
                    print(f"  输出形状: {test_output.shape}")
                    print(f"  输出范围: [{test_output.min():.4f}, {test_output.max():.4f}]")
                    
                    # 检查是否有非零输出
                    nonzero_ratio = (test_output != 0).float().mean().item()
                    print(f"  非零输出比例: {nonzero_ratio:.4f}")
                    
            except Exception as e:
                print(f"  测试失败: {e}")
        
        break
    
    print("\n" + "="*80)
    print("🔧 可能的解决方案:")
    print("="*80)
    print("1. ANN模式下IF层可能退化为identity函数，导致梯度为0")
    print("2. thresh参数可能需要特殊初始化")
    print("3. 可能需要在SNN模式下计算重要性")
    print("4. 可能需要修改重要性计算公式")


if __name__ == "__main__":
    diagnose_if_layer_issue() 