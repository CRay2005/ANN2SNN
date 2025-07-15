#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from Models import modelpool
from Preprocess import datapool
from utils import seed_all
import pandas as pd

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

def print_if_module_info(model):
    """打印所有IF模块的详细信息，包括阈值参数和梯度信息"""
    print("="*80)
    print("IF模块详细信息")
    print("="*80)
    
    from Models.layer import IF
    
    if_module_count = 0
    
    # 通过模块查找IF层
    for name, module in model.named_modules():
        if isinstance(module, IF):
            if_module_count += 1
            print(f"IF模块: {name}")
            print(f"  阈值(thresh): {module.thresh.item():.6f}")
            # print(f"  gamma参数: {module.gama}")
            # print(f"  时间步数(T): {module.T}")
            # print(f"  量化级别(L): {module.L}")
            
            # 打印阈值参数的梯度
            if module.thresh.grad is not None:
                thresh_grad = module.thresh.grad.item()
                print(f"  阈值梯度: {thresh_grad:.6f}")
            else:
                print(f"  阈值梯度: None")
            
            print("-"*60)
    
    # 总结
    if if_module_count == 0:
        print("未找到IF模块")
    else:
        print(f"总共找到 {if_module_count} 个IF模块")
    print("="*80)

def new_print_if_module_info(model):
    """打印所有IF模块的详细信息，包括输入梯度和输出梯度"""
    print("="*80)
    print("IF模块详细信息（包含输入输出梯度）")
    print("="*80)
    
    from Models.layer import IF
    
    if_module_count = 0
    
    # 存储梯度信息的字典
    gradient_info = {}
    
    # 为每个IF层注册钩子来捕获输入和输出梯度
    def register_if_hooks():
        hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, IF):
                # 存储该模块的梯度信息
                gradient_info[name] = {
                    'input_grad': None,
                    'output_grad': None,
                    'module': module
                }
                
                # 注册输出梯度钩子
                def create_output_hook(module_name):
                    def output_hook(module, grad_input, grad_output):
                        if grad_output[0] is not None:
                            gradient_info[module_name]['output_grad'] = grad_output[0].detach().clone()
                        
                        # 打印详细的梯度信息
                        print(f"\n{module_name} 梯度信息:")
                        print(f"  grad_output (dL/dy): {grad_output[0].shape if grad_output[0] is not None else 'None'}")
                        
                        # 针对IF层的特殊性：只有输入梯度，没有权重和偏置梯度
                        if isinstance(module, IF):
                            print(f"  grad_input (dL/dx): {[g.shape for g in grad_input if g is not None]}")
                            print(f"  📝 IF层说明: 只有输入梯度dL/dx，无权重梯度dL/dW和偏置梯度dL/db")
                        else:
                            print(f"  grad_input (dL/dx, dL/dW, dL/db): {[g.shape for g in grad_input if g is not None]}")
                        
                        # 详细分析grad_input的每个元素
                        for i, grad in enumerate(grad_input):
                            if grad is not None:
                                if isinstance(module, IF):
                                    print(f"    grad_input[{i}] (dL/dx): shape={grad.shape}, norm={grad.norm().item():.6f}, mean={grad.mean().item():.6f}")
                                else:
                                    if i == 0:
                                        grad_type = "dL/dx"
                                    elif i == 1:
                                        grad_type = "dL/dW"
                                    elif i == 2:
                                        grad_type = "dL/db"
                                    else:
                                        grad_type = f"dL/dparam{i}"
                                    print(f"    grad_input[{i}] ({grad_type}): shape={grad.shape}, norm={grad.norm().item():.6f}, mean={grad.mean().item():.6f}")
                            else:
                                if isinstance(module, IF):
                                    print(f"    grad_input[{i}] (dL/dx): None")
                                else:
                                    print(f"    grad_input[{i}]: None")
                        
                        # 分析grad_output
                        if grad_output[0] is not None:
                            print(f"  grad_output[0]: shape={grad_output[0].shape}, norm={grad_output[0].norm().item():.6f}, mean={grad_output[0].mean().item():.6f}")
                        
                        # 针对IF层，额外显示阈值梯度信息
                        if isinstance(module, IF) and module.thresh.grad is not None:
                            print(f"  🎯 IF层阈值梯度 (dL/dthresh): {module.thresh.grad.item():.6f}")
                        
                        print("-" * 40)
                    return output_hook
                
                # 注册输入梯度钩子
                def create_input_hook(module_name):
                    def input_hook(module, grad_input, grad_output):
                        if grad_input[0] is not None:
                            gradient_info[module_name]['input_grad'] = grad_input[0].detach().clone()
                    return input_hook
                
                output_hook = module.register_full_backward_hook(create_output_hook(name))
                input_hook = module.register_full_backward_hook(create_input_hook(name))
                hooks.extend([output_hook, input_hook])
        
        return hooks
    
    # 注册钩子
    hooks = register_if_hooks()
    
    try:
        # 通过模块查找IF层并打印信息
        for name, module in model.named_modules():
            if isinstance(module, IF):
                if_module_count += 1
                print(f"IF模块: {name}")
                print(f"  阈值(thresh): {module.thresh.item():.6f}")
                print(f"  时间步数(T): {module.T}")
                print(f"  量化级别(L): {module.L}")
                print(f"  代理梯度类型: {module.surrogate_grad}")
                print(f"  缩放因子: {module.scale}")
                
                # 打印阈值参数的梯度
                if module.thresh.grad is not None:
                    thresh_grad = module.thresh.grad.item()
                    thresh_grad_norm = module.thresh.grad.norm().item()
                    print(f"  阈值梯度: {thresh_grad:.6f}")
                    print(f"  阈值梯度范数: {thresh_grad_norm:.6f}")
                else:
                    print(f"  阈值梯度: None")
                
                # 打印输入梯度信息
                input_grad = gradient_info[name]['input_grad']
                if input_grad is not None:
                    print(f"  输入梯度:")
                    print(f"    形状: {list(input_grad.shape)}")
                    print(f"    范数: {input_grad.norm().item():.6f}")
                    print(f"    均值: {input_grad.mean().item():.6f}")
                    print(f"    标准差: {input_grad.std().item():.6f}")
                    print(f"    最小值: {input_grad.min().item():.6f}")
                    print(f"    最大值: {input_grad.max().item():.6f}")
                    
                    # 计算非零梯度比例
                    non_zero_ratio = (input_grad != 0).float().mean().item()
                    print(f"    非零梯度比例: {non_zero_ratio:.2%}")
                    
                    # 计算梯度分布
                    grad_abs = input_grad.abs()
                    print(f"    梯度分布:")
                    print(f"      25%分位数: {torch.quantile(grad_abs, 0.25).item():.6f}")
                    print(f"      50%分位数: {torch.quantile(grad_abs, 0.50).item():.6f}")
                    print(f"      75%分位数: {torch.quantile(grad_abs, 0.75).item():.6f}")
                    print(f"      95%分位数: {torch.quantile(grad_abs, 0.95).item():.6f}")
                else:
                    print(f"  输入梯度: None")
                
                # 打印输出梯度信息
                output_grad = gradient_info[name]['output_grad']
                if output_grad is not None:
                    print(f"  输出梯度:")
                    print(f"    形状: {list(output_grad.shape)}")
                    print(f"    范数: {output_grad.norm().item():.6f}")
                    print(f"    均值: {output_grad.mean().item():.6f}")
                    print(f"    标准差: {output_grad.std().item():.6f}")
                    print(f"    最小值: {output_grad.min().item():.6f}")
                    print(f"    最大值: {output_grad.max().item():.6f}")
                    
                    # 计算非零梯度比例
                    non_zero_ratio = (output_grad != 0).float().mean().item()
                    print(f"    非零梯度比例: {non_zero_ratio:.2%}")
                    
                    # 计算梯度分布
                    grad_abs = output_grad.abs()
                    print(f"    梯度分布:")
                    print(f"      25%分位数: {torch.quantile(grad_abs, 0.25).item():.6f}")
                    print(f"      50%分位数: {torch.quantile(grad_abs, 0.50).item():.6f}")
                    print(f"      75%分位数: {torch.quantile(grad_abs, 0.75).item():.6f}")
                    print(f"      95%分位数: {torch.quantile(grad_abs, 0.95).item():.6f}")
                else:
                    print(f"  输出梯度: None")
                
                print("-"*60)
        
        # 总结
        if if_module_count == 0:
            print("未找到IF模块")
        else:
            print(f"总共找到 {if_module_count} 个IF模块")
            print(f"已捕获输入和输出梯度信息")
    
    finally:
        # 清理钩子
        for hook in hooks:
            hook.remove()
    
    print("="*80)

def get_if_layer_input_output_gradients(model, dataloader, criterion):
    """获取IF层的输入和输出梯度（需要完整的前向和反向传播）"""
    print("="*80)
    print("IF层输入输出梯度分析")
    print("="*80)
    
    from Models.layer import IF
    
    # 存储梯度信息的字典
    gradient_info = {}
    
    # 为每个IF层注册钩子
    def register_gradient_hooks():
        hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, IF):
                gradient_info[name] = {
                    'input_grad': None,
                    'output_grad': None,
                    'module': module
                }
                
                # 注册输出梯度钩子
                def create_output_hook(module_name):
                    def output_hook(module, grad_input, grad_output):
                        if grad_output[0] is not None:
                            gradient_info[module_name]['output_grad'] = grad_output[0].detach().clone()

                        if grad_input[0] is not None:
                            gradient_info[module_name]['input_grad'] = grad_input[0].detach().clone()

                        # 打印详细的梯度信息
                        print(f"\n{module_name} 梯度信息:")
                        print(f"grad_output (dL/dy): {grad_output[0].shape if grad_output[0] is not None else 'None'}")
                        
                        # 针对IF层的特殊性：只有输入梯度，没有权重和偏置梯度
                        if isinstance(module, IF):
                            print(f"  grad_input (dL/dx): {[g.shape for g in grad_input if g is not None]}")
                            print(f"  📝 IF层说明: 只有输入梯度dL/dx，无权重梯度dL/dW和偏置梯度dL/db")
                        else:
                            print(f"  grad_input (dL/dx, dL/dW, dL/db): {[g.shape for g in grad_input if g is not None]}")
                        
                        # # 详细分析grad_input的每个元素
                        # for i, grad in enumerate(grad_input):
                        #     if grad is not None:
                        #         if isinstance(module, IF):
                        #             print(f"    grad_input[{i}] (dL/dx): shape={grad.shape}, norm={grad.norm().item():.6f}, mean={grad.mean().item():.6f}")
                        #         else:
                        #             if i == 0:
                        #                 grad_type = "dL/dx"
                        #             elif i == 1:
                        #                 grad_type = "dL/dW"
                        #             elif i == 2:
                        #                 grad_type = "dL/db"
                        #             else:
                        #                 grad_type = f"dL/dparam{i}"
                        #             print(f"    grad_input[{i}] ({grad_type}): shape={grad.shape}, norm={grad.norm().item():.6f}, mean={grad.mean().item():.6f}")
                        #     else:
                        #         if isinstance(module, IF):
                        #             print(f"    grad_input[{i}] (dL/dx): None")
                        #         else:
                        #             print(f"    grad_input[{i}]: None")
                        
                        # 分析grad_output
                        if grad_output[0] is not None:
                            print(f"  grad_output[0]: shape={grad_output[0].shape}, norm={grad_output[0].norm().item():.6f}, mean={grad_output[0].mean().item():.6f}")
                        
                        # 针对IF层，额外显示阈值梯度信息
                        if isinstance(module, IF) and module.thresh.grad is not None:
                            print(f"  🎯 IF层阈值梯度 (dL/dthresh): {module.thresh.grad.item():.6f}")
                        
                        print("-" * 40)
                    return output_hook
                
                hook = module.register_full_backward_hook(create_output_hook(name))
                hooks.append(hook)
        
        return hooks
    
    # 注册钩子
    hooks = register_gradient_hooks()
    
    try:
        # 确保模型处于训练模式
        model.train()
        
        # 获取一批数据
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
        
        # 清空梯度
        model.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 处理SNN输出
        if len(outputs.shape) > 2:
            outputs = outputs.mean(0)
        
        # 计算损失
        loss = criterion(outputs, targets)
        print(f"损失: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 分析每个IF层
        if_module_count = 0
        for name, module in model.named_modules():
            if isinstance(module, IF):
                if_module_count += 1
                print(f"IF模块: {name}")
                print(f"  阈值(thresh): {module.thresh.item():.6f}")
                print(f"  时间步数(T): {module.T}")
                print(f"  量化级别(L): {module.L}")
                
                # 打印阈值梯度
                if module.thresh.grad is not None:
                    thresh_grad = module.thresh.grad.item()
                    thresh_grad_norm = module.thresh.grad.norm().item()
                    print(f"  阈值梯度: {thresh_grad:.6f}")
                    print(f"  阈值梯度范数: {thresh_grad_norm:.6f}")
                else:
                    print(f"  阈值梯度: None")
                
                # 打印输出梯度
                output_grad = gradient_info[name]['output_grad']
                if output_grad is not None:
                    print(f"  输出梯度:")
                    print(f"    形状: {list(output_grad.shape)}")
                    print(f"    范数: {output_grad.norm().item():.6f}")
                    print(f"    均值: {output_grad.mean().item():.6f}")
                    print(f"    标准差: {output_grad.std().item():.6f}")
                    print(f"    最小值: {output_grad.min().item():.6f}")
                    print(f"    最大值: {output_grad.max().item():.6f}")
                    
                    # 计算非零梯度比例
                    non_zero_ratio = (output_grad != 0).float().mean().item()
                    print(f"    非零梯度比例: {non_zero_ratio:.2%}")
                    
                    # 计算梯度分布
                    grad_abs = output_grad.abs()
                    print(f"    梯度分布:")
                    print(f"      25%分位数: {torch.quantile(grad_abs, 0.25).item():.6f}")
                    print(f"      50%分位数: {torch.quantile(grad_abs, 0.50).item():.6f}")
                    print(f"      75%分位数: {torch.quantile(grad_abs, 0.75).item():.6f}")
                    print(f"      95%分位数: {torch.quantile(grad_abs, 0.95).item():.6f}")
                else:
                    print(f"  输出梯度: None")
                
                # 尝试获取输入梯度（通过检查输入张量的梯度）
                # 打印输入梯度信息
                input_grad = gradient_info[name]['input_grad']
                if input_grad is not None:
                    print(f"  输入梯度:")
                    print(f"    形状: {list(input_grad.shape)}")
                    print(f"    范数: {input_grad.norm().item():.6f}")
                    print(f"    均值: {input_grad.mean().item():.6f}")
                    print(f"    标准差: {input_grad.std().item():.6f}")
                    print(f"    最小值: {input_grad.min().item():.6f}")
                    print(f"    最大值: {input_grad.max().item():.6f}")
                    
                    # 计算非零梯度比例
                    non_zero_ratio = (input_grad != 0).float().mean().item()
                    print(f"    非零梯度比例: {non_zero_ratio:.2%}")
                    
                    # 计算梯度分布
                    grad_abs = input_grad.abs()
                    print(f"    梯度分布:")
                    print(f"      25%分位数: {torch.quantile(grad_abs, 0.25).item():.6f}")
                    print(f"      50%分位数: {torch.quantile(grad_abs, 0.50).item():.6f}")
                    print(f"      75%分位数: {torch.quantile(grad_abs, 0.75).item():.6f}")
                    print(f"      95%分位数: {torch.quantile(grad_abs, 0.95).item():.6f}")
                else:
                    print(f"  输入梯度: None")
                print("-"*60)
        
        # 总结
        if if_module_count == 0:
            print("未找到IF模块")
        else:
            print(f"总共找到 {if_module_count} 个IF模块")
            print(f"已成功捕获输出梯度信息")
    
    finally:
        # 清理钩子
        for hook in hooks:
            hook.remove()
    
    print("="*80)

class GradientAnalyzer:
    """全连接层梯度分析器（参考gradient_cray.py）"""
    def __init__(self, model):
        self.model = model
        self.gradient_hooks = {}
        self.gradient_records = {}
        
    def register_gradient_hooks(self):
        """为所有全连接层注册梯度记录钩子"""
        print("注册全连接层梯度钩子...")
        
        # 移除现有钩子
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}
        
        # 查找所有全连接层
        fc_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                fc_count += 1
                # 为权重参数注册梯度钩子
                hook = self._gradient_hook(name)
                handle = module.weight.register_hook(hook)
                self.gradient_hooks[name] = handle
                print(f"  - 注册钩子: {name} (输入={module.in_features}, 输出={module.out_features})")
        
        print(f"总共注册了 {fc_count} 个全连接层的梯度钩子")
        
    def _gradient_hook(self, name):
        """创建梯度钩子函数"""
        def hook(grad):
            # 确保梯度有效
            if grad is None:
                return
            
            # 计算每个输出神经元的平均梯度
            if grad.dim() > 1:
                # 全连接层: 对输入维度求平均
                neuron_grads = grad.abs().mean(dim=1)  # [out_features]
            else:
                # 1D情况
                neuron_grads = grad.abs()
            
            # 保存梯度统计信息
            self.gradient_records[name] = neuron_grads.detach().cpu()
        return hook
    
    def analyze_gradients(self, dataloader, criterion, num_batches=5):
        """
        分析全连接层梯度分布（参考gradient_cray.py）
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        num_batches - 分析批次数
        
        返回:
        gradient_stats - 梯度统计信息
        """
        print(f"\n开始分析 {num_batches} 个批次的梯度分布...")
        
        # 注册梯度钩子
        self.register_gradient_hooks()
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in self.gradient_hooks.keys():
            gradient_stats[name] = {'values': []}
        
        # 处理指定批次数据
        batch_count = 0
        data_iter = iter(dataloader)
        
        for batch_idx in range(num_batches):
            try:
                inputs, targets = next(data_iter)
                inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            except StopIteration:
                print(f"数据不足，只处理了 {batch_idx} 个批次")
                break
                
            # 清空梯度
            self.model.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)  # 对时间维度求平均
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播（触发梯度钩子）
            loss.backward()
            
            # 收集梯度数据
            for name, grads in self.gradient_records.items():
                if grads is not None:
                    gradient_stats[name]['values'].extend(grads.numpy())
            
            batch_count += 1
            print(f"  处理批次 {batch_count}/{num_batches}, 损失: {loss.item():.6f}")
        
        # 计算梯度统计
        print("\n计算梯度统计信息...")
        for name, stats in gradient_stats.items():
            if stats['values']:
                values = np.array(stats['values'])
                stats['mean'] = np.mean(values)
                stats['std'] = np.std(values)
                stats['min'] = np.min(values)
                stats['max'] = np.max(values)
                stats['median'] = np.median(values)
                stats['num_neurons'] = len(values)
                
                # 计算百分位数
                stats['p25'] = np.percentile(values, 25)
                stats['p75'] = np.percentile(values, 75)
                stats['p95'] = np.percentile(values, 95)
        
        return gradient_stats
    
    def get_low_gradient_neurons(self, gradient_stats, ratio=0.1):
        """
        识别低梯度神经元
        
        参数:
        gradient_stats - analyze_gradients返回的统计数据
        ratio - 要识别的神经元比例
        
        返回:
        low_gradient_neurons - 低梯度神经元列表
        """
        low_neurons = []
        
        # 处理每层的梯度统计
        for layer_name, stats in gradient_stats.items():
            if 'values' not in stats or not stats['values']:
                continue
                
            # 对梯度值排序
            grads = np.array(stats['values'])
            sorted_indices = np.argsort(grads)
            
            # 计算低梯度阈值
            num_low = int(len(grads) * ratio)
            
            # 收集低梯度神经元
            for idx in sorted_indices[:num_low]:
                low_neurons.append({
                    'layer': layer_name,
                    'neuron_index': idx,
                    'grad_value': grads[idx],
                    'grad_percentile': (np.searchsorted(np.sort(grads), grads[idx]) + 1) / len(grads)
                })
        
        return low_neurons
    
    def print_gradient_analysis(self, gradient_stats):
        """打印梯度分析结果"""
        print("="*80)
        print("全连接层梯度分布分析")
        print("="*80)
        
        if not gradient_stats:
            print("没有收集到梯度数据")
            return
        
        for layer_name, stats in gradient_stats.items():
            if not stats.get('values'):
                continue
                
            print(f"\n层: {layer_name}")
            print(f"  神经元数量: {stats['num_neurons']:,}")
            print(f"  梯度统计:")
            print(f"    均值: {stats['mean']:.8f}")
            print(f"    标准差: {stats['std']:.8f}")
            print(f"    最小值: {stats['min']:.8f}")
            print(f"    最大值: {stats['max']:.8f}")
            print(f"    中位数: {stats['median']:.8f}")
            print(f"  梯度分布:")
            print(f"    25%分位数: {stats['p25']:.8f}")
            print(f"    75%分位数: {stats['p75']:.8f}")
            print(f"    95%分位数: {stats['p95']:.8f}")
            print("-"*60)
        
        # 分析低梯度神经元
        print("\n低梯度神经元分析:")
        for ratio in [0.05, 0.1, 0.2]:
            low_neurons = self.get_low_gradient_neurons(gradient_stats, ratio)
            print(f"  梯度最低 {ratio*100:.1f}% 的神经元数量: {len(low_neurons)}")
            
            if low_neurons:
                # 按层分组统计
                layer_counts = {}
                for neuron in low_neurons:
                    layer = neuron['layer']
                    if layer not in layer_counts:
                        layer_counts[layer] = 0
                    layer_counts[layer] += 1
                
                for layer, count in layer_counts.items():
                    print(f"    {layer}: {count} 个")
        
        print("="*80)
    
    def cleanup_hooks(self):
        """清理梯度钩子"""
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}

class OutputRedirector:
    """输出重定向器，同时输出到控制台和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取VGG16参数和梯度')
    parser.add_argument('--batch_size', default=32, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--num_batches', default=5, type=int, help='梯度分析的批次数')

    
    args = parser.parse_args()
    
    # 设置输出重定向（默认保存到文件）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gradient_analysis_{args.mode}_{timestamp}.txt"
    output_redirector = OutputRedirector(filename)
    sys.stdout = output_redirector
    print(f"输出将保存到文件: {filename}")
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    print(f"设备: {device}, 随机种子: {args.seed}")
    print(f"分析模式: {args.mode}")
    print(f"梯度分析批次数: {args.num_batches}")
    
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
        
        # # 加载数据
        # print("加载CIFAR10数据集...")
        train_loader, test_loader = datapool('cifar10', args.batch_size)
        
        # # 获取一批数据
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        # print(f"输入形状: {images.shape}, 标签形状: {labels.shape}")
        
        # # 前向传播
        # print("执行前向传播...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # # 处理SNN输出
        if len(outputs.shape) > 2:
            outputs = outputs.mean(0)
        
        loss = criterion(outputs, labels)
        print(f"损失: {loss.item():.6f}")
        
        # # 反向传播
        # print("执行反向传播...")
        loss.backward()
        
        # # 只打印IF层信息
        # print_if_module_info(model)
        
        # # 打印IF层详细信息（包含输入输出梯度）
        # print("\n" + "="*80)
        # print("IF层详细信息（包含输入输出梯度）")
        # print("="*80)
        # new_print_if_module_info(model)
        
        # 获取IF层的输入输出梯度（需要完整的前向和反向传播）
        print("\n" + "="*80)
        print("IF层输入输出梯度完整分析")
        print("="*80)
        get_if_layer_input_output_gradients(model, train_loader, criterion)
        
        # # 专门分析IF层的梯度分布特征
        # print("\n" + "="*80)
        # print("IF层梯度分布特征详细分析")
        # print("="*80)
        # analyze_if_gradient_distribution(model, train_loader, criterion)
        
        return
        # 分析全连接层梯度（默认启用）
        print("\n" + "="*80)
        print("开始全连接层梯度分析")
        print("="*80)
        
        # 创建梯度分析器
        analyzer = GradientAnalyzer(model)
        
        try:
            # 分析梯度分布
            gradient_stats = analyzer.analyze_gradients(
                train_loader, 
                criterion, 
                num_batches=args.num_batches
            )
            
            # 打印分析结果
            analyzer.print_gradient_analysis(gradient_stats)
            
            # 保存各层梯度信息到CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for layer_name, stats in gradient_stats.items():
                if not stats.get('values'):
                    continue
                
                # 获取该层的神经元数量（总梯度数除以batch数）
                num_neurons = len(stats['values']) // args.num_batches
                
                # 重塑数据为 [num_neurons, num_batches] 的形状
                gradient_values = np.array(stats['values']).reshape(num_neurons, args.num_batches)
                
                # 计算每个神经元的平均梯度值
                mean_gradients = np.mean(gradient_values, axis=1)
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'neuron_index': range(num_neurons)
                })
                
                # 添加每个batch的梯度值列
                for i in range(args.num_batches):
                    df[f'gradient_batch_{i+1}'] = gradient_values[:, i]
                
                # 添加平均梯度值列
                df['gradient_mean'] = mean_gradients
                
                # 生成文件名
                filename = f"gradient_analysis_{layer_name}_{timestamp}.csv"
                
                # 保存到CSV
                df.to_csv(filename, index=False)
                print(f"已保存{layer_name}层的梯度信息到: {filename}")
                print(f"  神经元数量: {num_neurons}")
                print(f"  每个神经元包含{args.num_batches}个batch的梯度值和平均值")
            
            # 详细分析低梯度神经元
            print("\n" + "="*80)
            print("低梯度神经元详细分析")
            print("="*80)
            
            for ratio in [0.05, 0.1, 0.15, 0.2]:
                low_neurons = analyzer.get_low_gradient_neurons(gradient_stats, ratio)
                print(f"\n梯度最低 {ratio*100:.1f}% 的神经元详情:")
                
                if low_neurons:
                    # 按层分组显示
                    layer_groups = {}
                    for neuron in low_neurons:
                        layer = neuron['layer']
                        if layer not in layer_groups:
                            layer_groups[layer] = []
                        layer_groups[layer].append(neuron)
                    
                    for layer, neurons in layer_groups.items():
                        print(f"  {layer}: {len(neurons)} 个神经元")
                        # 显示前5个最低梯度的神经元
                        for i, neuron in enumerate(sorted(neurons, key=lambda x: x['grad_value'])[:5]):
                            print(f"    #{i+1}: 神经元{neuron['neuron_index']}, 梯度={neuron['grad_value']:.8f}, 百分位={neuron['grad_percentile']:.3f}")
                else:
                    print("  无数据")
            
        finally:
            # 清理梯度钩子
            analyzer.cleanup_hooks()
        
        print("\n✅ 完成!")
        print("\n💡 使用说明:")
        print("python 0612get_grad.py --mode snn  # SNN模式查看IF层信息和梯度分析")
        print("python 0612get_grad.py --mode ann  # ANN模式查看IF层信息和梯度分析")
        print("python 0612get_grad.py --mode snn --num_batches 10  # 指定分析批次数")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复标准输出并关闭文件
        if output_redirector is not None:
            sys.stdout = output_redirector.terminal
            output_redirector.close()
            print(f"输出已保存到: {filename}")

if __name__ == "__main__":
    main() 

def test_new_print_if_module_info():
    """测试new_print_if_module_info函数的功能"""
    print("="*80)
    print("测试new_print_if_module_info函数")
    print("="*80)
    
    try:
        # 创建模型
        model = modelpool('vgg16', 'cifar10')
        
        # 加载预训练模型
        model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 处理旧版本state_dict的兼容性
        keys = list(state_dict.keys())
        for k in keys:
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        
        # 设置为SNN模式
        model.set_T(8)
        model.set_L(4)
        model.train()
        
        # 创建测试数据
        test_input = torch.randn(1, 3, 32, 32, requires_grad=True)
        criterion = nn.CrossEntropyLoss()
        
        # 前向传播
        output = model(test_input)
        if len(output.shape) > 2:
            output = output.mean(0)
        
        # 计算损失
        target = torch.tensor([0])
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 测试新函数
        new_print_if_module_info(model)
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

# 如果要运行测试，取消下面的注释
# test_new_print_if_module_info() 

def analyze_if_gradient_distribution(model, dataloader, criterion):
    """专门分析IF层的梯度分布特征"""
    print("="*80)
    print("IF层梯度分布特征分析")
    print("="*80)
    
    from Models.layer import IF
    
    # 存储梯度信息的字典
    gradient_info = {}
    
    # 为每个IF层注册钩子
    def register_gradient_hooks():
        hooks = []
        
        for name, module in model.named_modules():
            if isinstance(module, IF):
                gradient_info[name] = {
                    'input_grad': None,
                    'output_grad': None,
                    'threshold_grad': None,
                    'module': module
                }
                
                # 注册输出梯度钩子
                def create_output_hook(module_name):
                    def output_hook(module, grad_input, grad_output):
                        if grad_output[0] is not None:
                            gradient_info[module_name]['output_grad'] = grad_output[0].detach().clone()
                        
                        # 捕获输入梯度（对于IF层，只有dL/dx）
                        if grad_input[0] is not None:
                            gradient_info[module_name]['input_grad'] = grad_input[0].detach().clone()
                        
                        # 捕获阈值梯度
                        if module.thresh.grad is not None:
                            gradient_info[module_name]['threshold_grad'] = module.thresh.grad.detach().clone()
                        
                    return output_hook
                
                hook = module.register_full_backward_hook(create_output_hook(name))
                hooks.append(hook)
        
        return hooks
    
    # 注册钩子
    hooks = register_gradient_hooks()
    
    try:
        # 确保模型处于训练模式
        model.train()
        
        # 获取一批数据
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
        
        # 清空梯度
        model.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 处理SNN输出
        if len(outputs.shape) > 2:
            outputs = outputs.mean(0)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 分析每个IF层的梯度分布
        for name, module in model.named_modules():
            if isinstance(module, IF):
                print(f"\n🔍 IF层: {name}")
                print("-" * 50)
                
                # 1. 阈值梯度分析
                thresh_grad = gradient_info[name]['threshold_grad']
                if thresh_grad is not None:
                    print(f"🎯 阈值梯度 (dL/dthresh):")
                    print(f"  数值: {thresh_grad.item():.8f}")
                    print(f"  绝对值: {abs(thresh_grad.item()):.8f}")
                    print(f"  符号: {'正' if thresh_grad.item() > 0 else '负' if thresh_grad.item() < 0 else '零'}")
                else:
                    print(f"🎯 阈值梯度: None")
                
                # 2. 输入梯度分析
                input_grad = gradient_info[name]['input_grad']
                if input_grad is not None:
                    print(f"\n📥 输入梯度 (dL/dx) 分布:")
                    print(f"  形状: {list(input_grad.shape)}")
                    print(f"  范数: {input_grad.norm().item():.6f}")
                    print(f"  均值: {input_grad.mean().item():.6f}")
                    print(f"  标准差: {input_grad.std().item():.6f}")
                    print(f"  最小值: {input_grad.min().item():.6f}")
                    print(f"  最大值: {input_grad.max().item():.6f}")
                    
                    # 梯度分布统计
                    grad_abs = input_grad.abs()
                    print(f"  绝对值分布:")
                    print(f"    25%分位数: {torch.quantile(grad_abs, 0.25).item():.6f}")
                    print(f"    50%分位数: {torch.quantile(grad_abs, 0.50).item():.6f}")
                    print(f"    75%分位数: {torch.quantile(grad_abs, 0.75).item():.6f}")
                    print(f"    90%分位数: {torch.quantile(grad_abs, 0.90).item():.6f}")
                    print(f"    95%分位数: {torch.quantile(grad_abs, 0.95).item():.6f}")
                    
                    # 非零梯度比例
                    non_zero_ratio = (input_grad != 0).float().mean().item()
                    print(f"  非零梯度比例: {non_zero_ratio:.2%}")
                    
                    # 梯度稀疏性分析
                    small_grad_ratio = (grad_abs < 0.01).float().mean().item()
                    print(f"  小梯度比例 (<0.01): {small_grad_ratio:.2%}")
                    
                    # 梯度方向分析
                    positive_ratio = (input_grad > 0).float().mean().item()
                    negative_ratio = (input_grad < 0).float().mean().item()
                    zero_ratio = (input_grad == 0).float().mean().item()
                    print(f"  梯度方向分布:")
                    print(f"    正值: {positive_ratio:.2%}")
                    print(f"    负值: {negative_ratio:.2%}")
                    print(f"    零值: {zero_ratio:.2%}")
                else:
                    print(f"📥 输入梯度: None")
                
                # 3. 输出梯度分析
                output_grad = gradient_info[name]['output_grad']
                if output_grad is not None:
                    print(f"\n📤 输出梯度 (dL/dy) 分布:")
                    print(f"  形状: {list(output_grad.shape)}")
                    print(f"  范数: {output_grad.norm().item():.6f}")
                    print(f"  均值: {output_grad.mean().item():.6f}")
                    print(f"  标准差: {output_grad.std().item():.6f}")
                    print(f"  最小值: {output_grad.min().item():.6f}")
                    print(f"  最大值: {output_grad.max().item():.6f}")
                    
                    # 输出梯度分布统计
                    out_grad_abs = output_grad.abs()
                    print(f"  绝对值分布:")
                    print(f"    25%分位数: {torch.quantile(out_grad_abs, 0.25).item():.6f}")
                    print(f"    50%分位数: {torch.quantile(out_grad_abs, 0.50).item():.6f}")
                    print(f"    75%分位数: {torch.quantile(out_grad_abs, 0.75).item():.6f}")
                    print(f"    90%分位数: {torch.quantile(out_grad_abs, 0.90).item():.6f}")
                    print(f"    95%分位数: {torch.quantile(out_grad_abs, 0.95).item():.6f}")
                    
                    # 非零梯度比例
                    non_zero_ratio = (output_grad != 0).float().mean().item()
                    print(f"  非零梯度比例: {non_zero_ratio:.2%}")
                else:
                    print(f"📤 输出梯度: None")
                
                # 4. 梯度传播效率分析
                if input_grad is not None and output_grad is not None:
                    input_norm = input_grad.norm().item()
                    output_norm = output_grad.norm().item()
                    if output_norm > 0:
                        propagation_ratio = input_norm / output_norm
                        print(f"\n🔄 梯度传播效率:")
                        print(f"  输入梯度范数: {input_norm:.6f}")
                        print(f"  输出梯度范数: {output_norm:.6f}")
                        print(f"  传播比例: {propagation_ratio:.6f}")
                        
                        if propagation_ratio < 0.1:
                            print(f"  ⚠️  警告: 梯度传播比例较低，可能存在梯度消失")
                        elif propagation_ratio > 10:
                            print(f"  ⚠️  警告: 梯度传播比例较高，可能存在梯度爆炸")
                        else:
                            print(f"  ✅ 梯度传播比例正常")
                
                print("-" * 50)
        
        print(f"\n📊 总结:")
        print(f"  分析了 {len([m for m in model.modules() if isinstance(m, IF)])} 个IF层")
        print(f"  每个IF层只有1个阈值参数，无权重和偏置参数")
        print(f"  梯度信息包括: 输入梯度(dL/dx)、输出梯度(dL/dy)、阈值梯度(dL/dthresh)")
    
    finally:
        # 清理钩子
        for hook in hooks:
            hook.remove()
    
    print("="*80) 

def test_analyze_if_gradient_distribution():
    """测试IF层梯度分布分析函数"""
    print("="*80)
    print("测试IF层梯度分布分析函数")
    print("="*80)
    
    import torch
    import torch.nn as nn
    from Models import modelpool
    from Preprocess import datapool
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 创建模型
        print("创建VGG16模型...")
        model = modelpool('vgg16', 'cifar10')
        
        # 加载预训练模型
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
        
        # 设置为SNN模式
        model.set_T(8)
        model.set_L(4)
        print("设置为SNN模式: T=8, L=4")
        
        model.to(device)
        model.train()
        
        # 加载数据
        print("加载CIFAR10数据集...")
        train_loader, _ = datapool('cifar10', 32)
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 运行梯度分布分析
        print("开始IF层梯度分布分析...")
        analyze_if_gradient_distribution(model, train_loader, criterion)
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

# 如果要运行测试，取消下面的注释
# test_analyze_if_gradient_distribution() 