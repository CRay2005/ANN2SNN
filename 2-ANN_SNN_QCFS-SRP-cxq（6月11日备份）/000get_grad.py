#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from Models import modelpool
from Preprocess import datapool
from utils import seed_all
from Models.layer import IF

# 导入000snngrad.py中的SNNGradientAnalyzer
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("snngrad", "000snngrad.py")
    snngrad_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snngrad_module)
    SNNGradientAnalyzer = snngrad_module.SNNGradientAnalyzer
except Exception as e:
    print(f"无法导入000snngrad.py: {e}")
    SNNGradientAnalyzer = None

class SNNSurrogateGradient:
    """SNN模拟梯度处理器，专门用于IF层梯度计算"""
    
    def __init__(self, model, temperature=5.0):
        self.model = model
        self.temperature = temperature
        self.if_activations = {}
        self.handles = []
        
    def register_if_hooks(self):
        """为IF层注册前向钩子以捕获激活值"""
        self.remove_hooks()
        
        for name, module in self.model.named_modules():
            if isinstance(module, IF):
                hook = self.make_if_forward_hook(name)
                handle = module.register_forward_hook(hook)
                self.handles.append(handle)
                
        print(f"已为 {len(self.handles)} 个IF层注册钩子")
    
    def make_if_forward_hook(self, layer_name):
        """创建IF层前向钩子"""
        def forward_hook(module, input, output):
            if len(input) > 0 and input[0] is not None:
                # 存储输入激活值（膜电位）
                self.if_activations[layer_name] = input[0].detach().clone()
        return forward_hook
    
    def apply_surrogate_gradient_to_if_layers(self):
        """为IF层手动应用模拟梯度"""
        applied_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, IF) and module.thresh.grad is not None:
                # 获取对应的激活值
                if name in self.if_activations:
                    activation = self.if_activations[name]
                    
                    # 计算模拟梯度
                    surrogate_grad = self.sigmoid_surrogate_grad(activation - module.thresh)
                    
                    # 计算激活值相对于阈值的影响
                    activation_effect = surrogate_grad.mean().item()
                    
                    # 修正阈值梯度
                    original_grad = module.thresh.grad.item()
                    modified_grad = original_grad * activation_effect
                    
                    # 应用修正后的梯度
                    module.thresh.grad.data.fill_(modified_grad)
                    
                    applied_count += 1
                    print(f"IF层 {name}: 原梯度={original_grad:.6f}, 修正梯度={modified_grad:.6f}, 激活效应={activation_effect:.6f}")
        
        return applied_count
    
    def sigmoid_surrogate_grad(self, x):
        """Sigmoid模拟梯度函数"""
        return torch.sigmoid(self.temperature * x) * (1 - torch.sigmoid(self.temperature * x))
    
    def triangular_surrogate_grad(self, x):
        """三角形模拟梯度函数"""
        return torch.clamp(1.0 - torch.abs(x), 0.0, 1.0)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

def get_params_grad(model):
    """
    获取模型参数和对应的梯度
    参考自hessian_weight_importance.py
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def print_params_and_gradients(model):
    """打印模型参数和梯度信息"""
    print("="*80)
    print("VGG16模型参数和梯度信息")
    print("="*80)
    
    params, grads = get_params_grad(model)
    
    print(f"总共有 {len(params)} 个需要梯度的参数")
    print("-"*80)
    
    total_params = 0
    total_grad_norm = 0.0
    
    for i, (param, grad) in enumerate(zip(params, grads)):
        param_count = param.numel()
        total_params += param_count
        
        # 梯度统计
        if torch.is_tensor(grad):
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            non_zero_elements = (grad != 0).sum().item()
            total_grad_norm += grad_norm ** 2
        else:
            grad_norm = grad_mean = grad_std = 0.0
            non_zero_elements = 0
        
        print(f"参数 {i+1:2d}: 形状={list(param.shape)}, 数量={param_count:,}")
        print(f"  参数统计: 均值={param.mean().item():.6f}, 标准差={param.std().item():.6f}")
        print(f"  梯度统计: 范数={grad_norm:.6f}, 均值={grad_mean:.6f}, 标准差={grad_std:.6f}")
        print(f"  非零梯度: {non_zero_elements:,}/{param_count:,} ({100*non_zero_elements/param_count:.2f}%)")
        print("-"*40)
    
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"\n总结: {total_params:,} 个参数, 总梯度范数: {total_grad_norm:.6f}")
    print("="*80)

def print_if_layers_only(model):
    """只打印IF层的参数和梯度信息"""
    print("="*80)
    print("IF层阈值参数和梯度信息")
    print("="*80)
    
    if_layer_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 只显示IF层的阈值参数
        if 'thresh' in name.lower() and param.numel() == 1:
            if_layer_count += 1
            print(f"IF层阈值: {name}")
            print(f"  形状: {list(param.shape)}, 参数数量: {param.numel():,}")
            print(f"  阈值: {param.data.item():.6f}")
            
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_value = param.grad.data.item()
                print(f"  梯度值: {grad_value:.6f}")
                print(f"  梯度范数: {grad_norm:.6f}, 梯度均值: {grad_mean:.6f}")
            else:
                print(f"  梯度: None")
            print("-"*60)
    
    if if_layer_count == 0:
        print("未找到IF层阈值参数")
    else:
        print(f"总共找到 {if_layer_count} 个IF层阈值参数")
    print("="*80)

def print_all_if_module_info(model):
    """打印所有IF模块的详细信息"""
    print("="*80)
    print("IF模块详细信息")
    print("="*80)
    
    from Models.layer import IF
    
    if_module_count = 0
    for name, module in model.named_modules():
        if isinstance(module, IF):
            if_module_count += 1
            print(f"IF模块: {name}")
            print(f"  阈值(thresh): {module.thresh.item():.6f}")
            print(f"  gamma参数: {module.gama}")
            print(f"  时间步数(T): {module.T}")
            print(f"  量化级别(L): {module.L}")
            
            # 打印阈值参数的梯度
            if module.thresh.grad is not None:
                thresh_grad = module.thresh.grad.item()
                print(f"  阈值梯度: {thresh_grad:.6f}")
            else:
                print(f"  阈值梯度: None")
            
            print("-"*60)
    
    if if_module_count == 0:
        print("未找到IF模块")
    else:
        print(f"总共找到 {if_module_count} 个IF模块")
    print("="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取VGG16参数和梯度')
    parser.add_argument('--batch_size', default=32, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')

    
    args = parser.parse_args()
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    print(f"设备: {device}, 随机种子: {args.seed}")
    
    try:
        # 创建模型
        print("创建VGG16模型...")
        model = modelpool('vgg16', 'cifar10')
        
        # 直接加载预训练模型
        model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
        print(f"加载预训练模型: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 处理旧版本state_dict的兼容性
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        print("✅ 预训练模型加载成功")
        
        if args.mode == 'snn':
            model.set_T(8)
            model.set_L(4)
            print("设置为SNN模式")
        else:
            model.set_T(0)
            print("设置为ANN模式")
        
        model.to(device)
        model.train()
        
        # 加载数据
        print("加载CIFAR10数据集...")
        train_loader, test_loader = datapool('cifar10', args.batch_size)
        
        # 获取一批数据
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        print(f"输入形状: {images.shape}, 标签形状: {labels.shape}")
        
        # 前向传播
        print("执行前向传播...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(images)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)
            
            loss = criterion(outputs, labels)
            print(f"损失: {loss.item():.6f}")
        except Exception as e:
            print(f"前向传播出错: {e}")
            if args.mode == 'snn':
                print("SNN模式可能存在时间维度问题，尝试降级到T=1")
                model.set_T(1)
                outputs = model(images)
                if len(outputs.shape) > 2:
                    outputs = outputs.mean(0)
                loss = criterion(outputs, labels)
                print(f"T=1模式损失: {loss.item():.6f}")
            else:
                raise e
        
        # 反向传播
        print("执行反向传播...")
        
        if args.mode == 'snn':
            # SNN模式：使用000snngrad.py中的模拟梯度
            print("SNN模式：使用模拟梯度反向传播")
            
            if SNNGradientAnalyzer is not None:
                # 创建SNN梯度分析器
                snn_analyzer = SNNGradientAnalyzer(model, surrogate_grad_type='sigmoid', grad_scale=5.0)
                
                # 使用模拟梯度进行反向传播
                print("使用模拟梯度进行反向传播...")
                try:
                    # 使用backward_with_surrogate方法
                    loss = snn_analyzer.backward_with_surrogate(outputs, labels, criterion)
                    print(f"✅ 模拟梯度反向传播完成，损失: {loss.item():.6f}")
                except Exception as e:
                    print(f"模拟梯度反向传播出错: {e}")
                    print("回退到标准反向传播")
                    model.zero_grad()  # 清空梯度避免重复反向传播
                    loss.backward(retain_graph=True)
                
                # 清理钩子
                if hasattr(snn_analyzer, 'gradient_hooks'):
                    for handle in snn_analyzer.gradient_hooks.values():
                        handle.remove()
                        
                print("✅ 完成SNN模拟梯度处理")
            else:
                print("⚠️ 无法导入SNNGradientAnalyzer，使用标准反向传播")
                loss.backward()
        else:
            # ANN模式：标准反向传播
            loss.backward()
        
        # 只打印IF层信息
        print_if_layers_only(model)
        print_all_if_module_info(model)
        
        print("✅ 完成!")
        print("\n💡 使用说明:")
        print("python 000get_grad.py --mode snn  # SNN模式查看IF层信息")
        print("python 000get_grad.py --mode ann  # ANN模式查看IF层信息")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 