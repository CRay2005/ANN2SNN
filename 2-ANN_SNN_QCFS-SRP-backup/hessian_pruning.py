#!/usr/bin/env python3
"""
基于Hessian Trace的权重重要性计算模块
实现公式: weight_importance = hessian_trace * (weight_norm^2 / num_weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class HessianWeightImportance:
    """
    Hessian权重重要性计算器
    
    主要功能：
    1. 计算真实的Hessian trace（小规模网络）
    2. Fisher信息矩阵近似（大规模网络）
    3. KFAC（Kronecker-Factored Approximate Curvature）近似
    4. 权重重要性评估和剪枝决策
    """
    
    def __init__(self, model, use_approximation='fisher', sample_size=100):
        self.model = model
        self.use_approximation = use_approximation
        self.sample_size = sample_size
        
        # 存储权重层和对应的重要性
        self.weight_layers = {}
        self.weight_importance = {}
        self.hessian_trace = {}
        self.gradient_buffer = defaultdict(list)
        
        # 注册权重层
        self._register_weight_layers()
        
    def _register_weight_layers(self):
        """注册所有权重层（Conv2d和Linear）"""
        layer_id = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_id += 1
                self.weight_layers[f'layer_{layer_id}'] = {
                    'name': name,
                    'module': module,
                    'weight_shape': module.weight.shape,
                    'num_weights': module.weight.numel()
                }
                print(f"注册权重层 {layer_id}: {name} - {module.weight.shape}")
    
    def compute_exact_hessian_trace(self, data_loader, criterion):
        """
        计算精确的Hessian trace（仅适用于小规模问题）
        
        Hessian trace = tr(∇²L) = Σᵢ ∂²L/∂wᵢ²
        """
        print("计算精确Hessian trace...")
        
        # 清零梯度
        self.model.zero_grad()
        
        total_samples = 0
        hessian_trace_sum = defaultdict(float)
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= self.sample_size // data.size(0):
                break
                
            data = data.cuda() if torch.cuda.is_available() else data
            target = target.cuda() if torch.cuda.is_available() else target
            
            # 前向传播
            output = self.model(data)
            if len(output.shape) > 2:  # SNN模式：[T, B, classes]
                output = output.mean(0)
            
            loss = criterion(output, target)
            
            # 计算一阶梯度
            first_grads = torch.autograd.grad(loss, self.model.parameters(), 
                                            create_graph=True, retain_graph=True)
            
            # 计算二阶梯度（Hessian对角线）
            for layer_name, layer_info in self.weight_layers.items():
                module = layer_info['module']
                
                # 找到对应的梯度
                param_idx = list(self.model.parameters()).index(module.weight)
                grad = first_grads[param_idx]
                
                # 计算Hessian对角线元素
                hessian_diag = []
                for i in range(grad.numel()):
                    grad_i = grad.flatten()[i]
                    if grad_i.requires_grad:
                        second_grad = torch.autograd.grad(grad_i, module.weight, 
                                                        retain_graph=True)[0]
                        hessian_diag.append(second_grad.flatten()[i].item())
                    else:
                        hessian_diag.append(0.0)
                
                # 计算trace
                trace = sum(hessian_diag)
                hessian_trace_sum[layer_name] += trace
                
            total_samples += data.size(0)
            
            # 清理计算图
            loss.backward(retain_graph=False)
            self.model.zero_grad()
        
        # 平均化
        for layer_name in hessian_trace_sum:
            self.hessian_trace[layer_name] = hessian_trace_sum[layer_name] / total_samples
            
        return self.hessian_trace
    
    def compute_fisher_approximation(self, data_loader, criterion):
        """
        使用Fisher信息矩阵近似Hessian
        
        Fisher信息矩阵: F = E[∇log p(y|x) ∇log p(y|x)ᵀ]
        Hessian ≈ Fisher (在最优点附近)
        """
        print("计算Fisher信息矩阵近似...")
        
        fisher_trace = defaultdict(float)
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                if batch_idx >= self.sample_size // data.size(0):
                    break
                    
                data = data.cuda() if torch.cuda.is_available() else data
                target = target.cuda() if torch.cuda.is_available() else target
                
                # 前向传播
                output = self.model(data)
                if len(output.shape) > 2:  # SNN模式
                    output = output.mean(0)
                
                # 计算概率
                probs = F.softmax(output, dim=1)
                
                # 计算Fisher信息（对角线近似）
                for layer_name, layer_info in self.weight_layers.items():
                    module = layer_info['module']
                    
                    # 使用激活值方差作为Fisher信息的代理
                    if hasattr(module, 'last_input'):
                        activations = module.last_input
                        
                        # 计算激活值的方差作为敏感性指标
                        if len(activations.shape) > 2:
                            variance = torch.var(activations, dim=[0, 2, 3] if len(activations.shape) == 4 else [0, 1])
                        else:
                            variance = torch.var(activations, dim=0)
                        
                        fisher_trace[layer_name] += variance.sum().item()
                
                total_samples += data.size(0)
        
        # 平均化
        for layer_name in fisher_trace:
            self.hessian_trace[layer_name] = fisher_trace[layer_name] / total_samples
            
        return self.hessian_trace
    
    def compute_kfac_approximation(self, data_loader, criterion):
        """
        KFAC (Kronecker-Factored Approximate Curvature) 近似
        
        对于卷积层: H ≈ A ⊗ S
        其中 A 是输入协方差，S 是梯度协方差
        """
        print("计算KFAC近似...")
        
        kfac_trace = defaultdict(float)
        activation_cov = defaultdict(list)
        gradient_cov = defaultdict(list)
        
        # 注册钩子收集激活值
        hooks = []
        
        def activation_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input = input[0]
                activation_cov[name].append(input.detach())
            return hook
        
        # 为每个权重层注册钩子
        for layer_name, layer_info in self.weight_layers.items():
            module = layer_info['module']
            hook = module.register_forward_hook(activation_hook(layer_name))
            hooks.append(hook)
        
        self.model.train()
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= self.sample_size // data.size(0):
                break
                
            data = data.cuda() if torch.cuda.is_available() else data
            target = target.cuda() if torch.cuda.is_available() else target
            
            self.model.zero_grad()
            
            output = self.model(data)
            if len(output.shape) > 2:
                output = output.mean(0)
            
            loss = criterion(output, target)
            loss.backward()
            
            # 收集梯度
            for layer_name, layer_info in self.weight_layers.items():
                module = layer_info['module']
                if module.weight.grad is not None:
                    gradient_cov[layer_name].append(module.weight.grad.detach().clone())
            
            total_samples += data.size(0)
        
        # 计算KFAC近似的trace
        for layer_name in self.weight_layers:
            if layer_name in activation_cov and layer_name in gradient_cov:
                # 简化版KFAC：使用激活值和梯度的方差乘积
                act_vars = [torch.var(act).item() for act in activation_cov[layer_name]]
                grad_vars = [torch.var(grad).item() for grad in gradient_cov[layer_name]]
                
                avg_act_var = sum(act_vars) / len(act_vars) if act_vars else 0
                avg_grad_var = sum(grad_vars) / len(grad_vars) if grad_vars else 0
                
                kfac_trace[layer_name] = avg_act_var * avg_grad_var
        
        # 清理钩子
        for hook in hooks:
            hook.remove()
        
        self.hessian_trace.update(kfac_trace)
        return self.hessian_trace
    
    def compute_weight_importance(self):
        """
        计算权重重要性
        
        weight_importance = hessian_trace * (weight_norm^2 / num_weights)
        """
        print("计算权重重要性...")
        
        for layer_name, layer_info in self.weight_layers.items():
            module = layer_info['module']
            
            # 获取权重范数
            weight_norm_sq = torch.norm(module.weight, p=2) ** 2
            num_weights = layer_info['num_weights']
            
            # 获取Hessian trace
            hessian_trace = self.hessian_trace.get(layer_name, 0.0)
            
            # 计算重要性
            importance = hessian_trace * (weight_norm_sq.item() / num_weights)
            
            self.weight_importance[layer_name] = {
                'importance': importance,
                'hessian_trace': hessian_trace,
                'weight_norm_sq': weight_norm_sq.item(),
                'num_weights': num_weights,
                'weight_shape': layer_info['weight_shape']
            }
            
            print(f"{layer_name}: importance={importance:.6f}, "
                  f"hessian_trace={hessian_trace:.6f}, "
                  f"weight_norm_sq={weight_norm_sq.item():.6f}")
        
        return self.weight_importance
    
    def get_pruning_mask(self, pruning_ratio=0.3, layer_type='all'):
        """
        基于权重重要性生成剪枝掩码
        
        Args:
            pruning_ratio: 剪枝比例
            layer_type: 'conv', 'fc', 'all'
        """
        if not self.weight_importance:
            print("请先计算权重重要性！")
            return {}
        
        # 过滤层类型
        filtered_importance = {}
        for layer_name, importance_info in self.weight_importance.items():
            layer_info = self.weight_layers[layer_name]
            module = layer_info['module']
            
            if layer_type == 'conv' and isinstance(module, nn.Conv2d):
                filtered_importance[layer_name] = importance_info['importance']
            elif layer_type == 'fc' and isinstance(module, nn.Linear):
                filtered_importance[layer_name] = importance_info['importance']
            elif layer_type == 'all':
                filtered_importance[layer_name] = importance_info['importance']
        
        if not filtered_importance:
            return {}
        
        # 排序并选择要剪枝的层
        sorted_layers = sorted(filtered_importance.items(), key=lambda x: x[1])
        num_layers_to_prune = int(len(sorted_layers) * pruning_ratio)
        
        pruning_mask = {}
        for i, (layer_name, importance) in enumerate(sorted_layers):
            if i < num_layers_to_prune:
                pruning_mask[layer_name] = True  # 标记为剪枝
                print(f"剪枝层: {layer_name}, importance={importance:.6f}")
            else:
                pruning_mask[layer_name] = False
        
        return pruning_mask
    
    def run_analysis(self, data_loader, criterion, pruning_ratio=0.3):
        """
        运行完整的Hessian权重重要性分析
        """
        print("="*60)
        print("开始Hessian权重重要性分析")
        print("="*60)
        
        # 计算Hessian trace
        if self.use_approximation == 'exact':
            self.compute_exact_hessian_trace(data_loader, criterion)
        elif self.use_approximation == 'fisher':
            self.compute_fisher_approximation(data_loader, criterion)
        elif self.use_approximation == 'kfac':
            self.compute_kfac_approximation(data_loader, criterion)
        
        # 计算权重重要性
        self.compute_weight_importance()
        
        # 生成剪枝掩码
        pruning_mask = self.get_pruning_mask(pruning_ratio)
        
        print("="*60)
        print("Hessian权重重要性分析完成")
        print("="*60)
        
        return {
            'hessian_trace': self.hessian_trace,
            'weight_importance': self.weight_importance,
            'pruning_mask': pruning_mask
        }


def register_activation_hooks(model):
    """为模型注册激活值收集钩子"""
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        module.last_input = input.detach()
        
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    return hooks


def remove_hooks(hooks):
    """移除钩子"""
    for hook in hooks:
        hook.remove() 