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
import warnings
import random
# IF 层用于记录 critical_count
from Models.layer import IF
# from Models.layer import load_model_compatible

# 设置环境变量抑制cuDNN警告
os.environ['CUDNN_V8_API_DISABLED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
# 抑制PyTorch相关警告
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class GradientAnalyzer:
    """全连接层梯度分析器"""
    def __init__(self, model):
        self.model = model
        self.gradient_hooks = {}
        self.gradient_records = {}
        self.weight_grad_hooks = {}
        self.tensor_grad_hooks = {}
        self.if_grad_hooks = {}


    def register_comprehensive_hooks(self):
        """注册梯度钩子"""
        # 移除现有钩子
        for handle in self.weight_grad_hooks.values():
            handle.remove()
        for handle in self.tensor_grad_hooks.values():
            handle.remove()
        for handle in self.if_grad_hooks.values():
            handle.remove()
        self.weight_grad_hooks = {}
        self.tensor_grad_hooks = {}
        self.if_grad_hooks = {}
        # 使用不同的变量名避免与register_gradient_hooks冲突
        self.comprehensive_gradient_records = {}
        
        # 查找所有全连接层和目标IF层
        fc_count = 0
        if_count = 0
        target_if_layers = ['classifier.2', 'classifier.5']
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                fc_count += 1
                
                # 1. 权重梯度钩子
                weight_hook = self._weight_gradient_hook(name)
                weight_handle = module.weight.register_hook(weight_hook)
                self.weight_grad_hooks[name] = weight_handle
                
                # 2. 张量梯度钩子（grad_input和grad_output）
                tensor_hook = self._tensor_gradient_hook(name)
                tensor_handle = module.register_full_backward_hook(tensor_hook)
                self.tensor_grad_hooks[name] = tensor_handle
                
                # 初始化记录
                self.comprehensive_gradient_records[name] = {
                    'layer_type': 'fc',
                    'weight_grad': None,      # 权重梯度
                    'input_grad': None,       # 输入梯度
                    'output_grad': None,      # 输出梯度
                    'importance_scores': {}   # 综合重要性分数
                }
            
            # 添加IF层梯度采集 (仅针对classifier.2和classifier.5)
            elif isinstance(module, IF) and name in target_if_layers:
                if_count += 1
                
                # 1. 阈值梯度钩子
                thresh_hook = self._threshold_gradient_hook(name)
                thresh_handle = module.thresh.register_hook(thresh_hook)
                self.if_grad_hooks[f"{name}_thresh"] = thresh_handle
                
                # 2. IF层张量梯度钩子
                if_tensor_hook = self._if_tensor_gradient_hook(name)
                if_tensor_handle = module.register_full_backward_hook(if_tensor_hook)
                self.if_grad_hooks[f"{name}_tensor"] = if_tensor_handle
                
                # 初始化IF层记录
                self.comprehensive_gradient_records[name] = {
                    'layer_type': 'if',
                    'threshold_grad': None,   # 阈值梯度
                    'input_grad': None,       # 输入梯度
                    'output_grad': None,      # 输出梯度
                    'threshold_value': None,  # 阈值数值
                }
        
        print(f"总共注册了 {fc_count} 个全连接层和 {if_count} 个IF层的梯度钩子")

    def _weight_gradient_hook(self, name):
        """权重梯度钩子"""
        def hook(grad):
            if grad is not None:
                # 计算每个输出神经元的平均权重梯度
                if grad.dim() > 1:
                    neuron_weight_grads = grad.abs().mean(dim=1)  # [out_features]
                else:
                    neuron_weight_grads = grad.abs()

                self.comprehensive_gradient_records[name]['weight_grad'] = neuron_weight_grads.detach().cpu()
        return hook
    
    def _tensor_gradient_hook(self, name):
        """张量梯度钩子"""
        def hook(module, grad_input, grad_output):

            
            # 捕获输入梯度
            if grad_input[0] is not None:
                input_grad = grad_input[0]  # [batch_size, in_features]
                # 计算每个输入神经元的平均梯度
                neuron_input_grads = input_grad.abs().mean(dim=0)  # [in_features]
                self.comprehensive_gradient_records[name]['input_grad'] = neuron_input_grads.detach().cpu()
            
            # 捕获输出梯度
            if grad_output[0] is not None:
                output_grad = grad_output[0]  # [batch_size, out_features]
                # 计算每个输出神经元的平均梯度
                neuron_output_grads = output_grad.abs().mean(dim=0)  # [out_features]
                self.comprehensive_gradient_records[name]['output_grad'] = neuron_output_grads.detach().cpu()
        return hook
    
    def _threshold_gradient_hook(self, name):
        """IF层阈值梯度钩子"""
        def hook(grad):
            if grad is not None:
                # 阈值梯度现在是向量，需要计算平均值
                thresh_grad = grad.abs().mean().item()  # 计算平均梯度
                self.comprehensive_gradient_records[name]['threshold_grad'] = thresh_grad
        return hook
    
    def _if_tensor_gradient_hook(self, name):
        """IF层张量梯度钩子"""
        def hook(module, grad_input, grad_output):
            
            # 保存阈值数值（现在是向量，计算平均值）
            self.comprehensive_gradient_records[name]['threshold_value'] = module.thresh.data.mean().item()
            
            # 捕获输入梯度
            if grad_input[0] is not None:
                input_grad = grad_input[0]  # [batch_size, features] or [T*batch_size, features]
                # 处理SNN模式的时间维度
                if module.T > 0:
                    # SNN模式: reshape回 [T, batch_size, features] 然后在时间维度平均
                    batch_size = input_grad.shape[0] // module.T
                    input_grad = input_grad.view(module.T, batch_size, -1)
                    input_grad = input_grad.mean(dim=0)  # 时间维度平均: [batch_size, features]
                
                # 计算每个神经元的平均梯度
                neuron_input_grads = input_grad.abs().mean(dim=0)  # [features]
                self.comprehensive_gradient_records[name]['input_grad'] = neuron_input_grads.detach().cpu()
            
            # 捕获输出梯度
            if grad_output[0] is not None:
                output_grad = grad_output[0]  # [batch_size, features] or [T*batch_size, features]
                # 处理SNN模式的时间维度
                if module.T > 0:
                    # SNN模式: reshape回 [T, batch_size, features] 然后在时间维度平均
                    batch_size = output_grad.shape[0] // module.T
                    output_grad = output_grad.view(module.T, batch_size, -1)
                    output_grad = output_grad.mean(dim=0)  # 时间维度平均: [batch_size, features]
                
                # 计算每个神经元的平均梯度
                neuron_output_grads = output_grad.abs().mean(dim=0)  # [features]
                self.comprehensive_gradient_records[name]['output_grad'] = neuron_output_grads.detach().cpu()
        return hook
    
    def register_gradient_hooks(self):
        """为所有全连接层注册梯度记录钩子"""
        # print("注册全连接层梯度钩子...")
        
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
                # print(f"  - 注册钩子: {name} (输入={module.in_features}, 输出={module.out_features})")
        
        # print(f"总共注册了 {fc_count} 个全连接层的梯度钩子")
        
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

    def analyze_gradients_improved(self, dataloader, criterion, num_batches=5):
        """
        分析全连接层梯度分布
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        num_batches - 分析批次数
        
        返回:
        gradient_stats - 梯度统计信息，包含三种梯度的平均值
        """
        print(f"\n开始分析 {num_batches} 个批次的梯度分布...")
        
        # 保存原始训练状态
        original_training = self.model.training
        
        # 注册梯度钩子
        self.register_comprehensive_hooks()
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in self.comprehensive_gradient_records.keys():
            layer_type = self.comprehensive_gradient_records[name]['layer_type']
            if layer_type == 'fc':
                gradient_stats[name] = {
                    'layer_type': 'fc',
                    'weight_grad_values': [],
                    'input_grad_values': [],
                    'output_grad_values': []
                }
            elif layer_type == 'if':
                gradient_stats[name] = {
                    'layer_type': 'if',
                    'threshold_grad_values': [],
                    'input_grad_values': [],
                    'output_grad_values': [],
                    'threshold_values': []
                }
        
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
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            
            # 反向传播（触发梯度钩子）
            loss.backward()
            
            # 收集梯度数据
            for name, records in self.comprehensive_gradient_records.items():
                layer_type = records['layer_type']
                
                if layer_type == 'fc':
                    # FC层梯度收集
                    if records['weight_grad'] is not None:
                        gradient_stats[name]['weight_grad_values'].append(records['weight_grad'].numpy())
                    if records['input_grad'] is not None:
                        gradient_stats[name]['input_grad_values'].append(records['input_grad'].numpy())
                    if records['output_grad'] is not None:
                        gradient_stats[name]['output_grad_values'].append(records['output_grad'].numpy())
                
                elif layer_type == 'if':
                    # IF层梯度收集
                    if records['threshold_grad'] is not None:
                        gradient_stats[name]['threshold_grad_values'].append(records['threshold_grad'])
                    if records['input_grad'] is not None:
                        gradient_stats[name]['input_grad_values'].append(records['input_grad'].numpy())
                    if records['output_grad'] is not None:
                        gradient_stats[name]['output_grad_values'].append(records['output_grad'].numpy())
                    if records['threshold_value'] is not None:
                        gradient_stats[name]['threshold_values'].append(records['threshold_value'])
            
            batch_count += 1
            print(f"  处理批次 {batch_count}/{num_batches}, 损失: {loss.item():.6f}, 准确率: {accuracy:.2f}%")
        
        # 计算平均梯度
        for name in gradient_stats:
            layer_type = gradient_stats[name]['layer_type']
            
            if layer_type == 'fc':
                # FC层平均梯度计算
                if gradient_stats[name]['weight_grad_values']:
                    gradient_stats[name]['weight_grad_values'] = np.mean(gradient_stats[name]['weight_grad_values'], axis=0)
                if gradient_stats[name]['input_grad_values']:
                    gradient_stats[name]['input_grad_values'] = np.mean(gradient_stats[name]['input_grad_values'], axis=0)
                if gradient_stats[name]['output_grad_values']:
                    gradient_stats[name]['output_grad_values'] = np.mean(gradient_stats[name]['output_grad_values'], axis=0)
            
            elif layer_type == 'if':
                # IF层平均梯度计算
                if gradient_stats[name]['threshold_grad_values']:
                    gradient_stats[name]['threshold_grad_values'] = np.mean(gradient_stats[name]['threshold_grad_values'])
                if gradient_stats[name]['input_grad_values']:
                    gradient_stats[name]['input_grad_values'] = np.mean(gradient_stats[name]['input_grad_values'], axis=0)
                if gradient_stats[name]['output_grad_values']:
                    gradient_stats[name]['output_grad_values'] = np.mean(gradient_stats[name]['output_grad_values'], axis=0)
                if gradient_stats[name]['threshold_values']:
                    gradient_stats[name]['threshold_values'] = np.mean(gradient_stats[name]['threshold_values'])
        
        # 恢复原始训练状态
        if original_training:
            self.model.train()
        else:
            self.model.eval()
        
        return gradient_stats

    def analyze_gradients(self, dataloader, criterion, num_batches=5):
        """
        分析全连接层梯度分布
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        num_batches - 分析批次数
        
        返回:
        gradient_stats - 梯度统计信息
        """
        # print(f"\n开始分析 {num_batches} 个批次的梯度分布...")
        
        # 注册梯度钩子
        self.register_gradient_hooks()
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in self.gradient_hooks.keys():
            gradient_stats[name] = {'values': None}
        
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
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy = 100 * correct / total
            
            # 反向传播（触发梯度钩子）
            loss.backward()
            
            # 收集梯度数据
            for name, grads in self.gradient_records.items():
                if grads is not None:
                    if gradient_stats[name]['values'] is None:
                        # 第一个批次，直接赋值
                        gradient_stats[name]['values'] = grads.numpy()
                    else:
                        # 后续批次，累加梯度
                        gradient_stats[name]['values'] += grads.numpy()
            
            batch_count += 1
        
        # 计算平均梯度
        for name in gradient_stats:
            if gradient_stats[name]['values'] is not None:
                gradient_stats[name]['values'] = gradient_stats[name]['values'] / batch_count
        
        # 计算梯度统计
        # print("\n计算梯度统计信息...")
        for name, stats in gradient_stats.items():
            if stats['values'] is not None:
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

    def get_low_gradient_neurons(self, gradient_stats, order='low', ratio=0.1, sort_by='gradient'):
        """
        识别低梯度神经元
        
        参数:
        gradient_stats - analyze_gradients返回的统计数据
        order - 排序方式: 'low'(从小到大), 'high'(从大到小), 'index'(按神经元序号), 'random'(随机)
        ratio - 要识别的神经元比例
        sort_by - 排序依据: 'gradient'(按梯度), 'weight'(按权重), 'weight_gradient'(按梯度*权重)
        
        返回:
        low_gradient_neurons - 低梯度神经元列表
        """
        low_neurons = []
        
        # 处理每层的梯度统计
        for layer_name, stats in gradient_stats.items():
            if layer_name == 'classifier.7':     # 跳过最后一层
                continue
            if 'values' not in stats or stats['values'] is None:
                continue
                
            # 获取梯度值
            grads = np.array(stats['values'])
            
            # 根据排序依据选择排序数据
            if sort_by == 'gradient':
                sort_data = grads
            elif sort_by == 'weight':
                # 获取权重数据
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_data = module.weight.data
                        # 计算每个神经元的平均权重绝对值
                        sort_data = weight_data.abs().mean(dim=1).cpu().numpy()
                        break
                else:
                    print(f"警告: 未找到层 {layer_name} 的权重数据，使用梯度排序")
                    sort_data = grads
            elif sort_by == 'weight_gradient':
                # 获取权重数据并计算权重*梯度
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_data = module.weight.data
                        # 计算每个神经元的平均权重绝对值
                        weights = weight_data.abs().mean(dim=1).cpu().numpy()
                        # 计算权重*梯度
                        sort_data = weights * grads
                        break
                else:
                    print(f"警告: 未找到层 {layer_name} 的权重数据，使用梯度排序")
                    sort_data = grads
            else:
                print(f"警告: 未知的排序依据 '{sort_by}'，使用梯度排序")
                sort_data = grads
            
            # 根据排序方式对数据进行排序
            if order == 'low' or order == 'combine':
                sorted_indices = np.argsort(sort_data)  # 从小到大排序
            elif order == 'high':
                sorted_indices = np.argsort(sort_data)[::-1]  # 从大到小排序
            elif order == 'index':
                sorted_indices = np.arange(len(sort_data))  # 按神经元序号排序
            elif order == 'random':
                # 随机打乱索引
                sorted_indices = np.random.permutation(len(sort_data))
            else :
                # 默认按从小到大排序
                sorted_indices = np.argsort(sort_data)
            
            # 计算要选择的神经元数量
            ratio1 = 0.1 
            ratio2 = 0.1
            if order == 'combine':
                num_select = int(len(sort_data) * ratio1)
            else:
                num_select = int(len(sort_data) * ratio)
            
            # 收集神经元信息
            for idx in sorted_indices[:num_select]:
                # 获取权重信息
                weight_info = 0.0
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_data = module.weight.data
                        weight_info = weight_data.abs().mean(dim=1).cpu().numpy()[idx]
                        break
                
                # 计算权重*梯度
                weight_gradient = weight_info * grads[idx]
                
                low_neurons.append({
                    'layer': layer_name,
                    'neuron_index': idx,
                    'grad_value': grads[idx],
                    'weight_value': weight_info,
                    'weight_gradient_value': weight_gradient,
                    'sort_value': sort_data[idx],  # 用于排序的值
                    'sort_by': sort_by,  # 排序依据
                    'grad_percentile': (np.searchsorted(np.sort(grads), grads[idx]) + 1) / len(grads)
                })
            if order == 'combine':
                low_neurons.sort(key=lambda x: x['sort_value'], reverse=True)
                low_neurons = low_neurons[:int(len(sort_data) * ratio2)]
        return low_neurons

    def get_selected_neurons(self, gradient_stats, order='low', ratio=0.1, sort_by='improved_method'):
        """
        基于改进的神经元重要性评估方法进行神经元选取
        
        参数:
        gradient_stats - analyze_gradients_improved返回的统计数据
        order - 排序方式: 'low'(从小到大), 'high'(从大到小), 'index'(按神经元序号), 'random'(随机)
        ratio - 要选择的神经元比例
        sort_by - 排序依据: 'improved_method'(改进方法)
        
        返回:
        neurons_to_prune - 要剪枝的神经元列表
        """
        neurons_to_prune = []
        
        # FC层与IF层的对应关系映射
        fc_to_if_mapping = {
            'classifier.1': 'classifier.2',  # FC层 -> 对应的IF层
            'classifier.4': 'classifier.5',  # FC层 -> 对应的IF层
            # classifier.6 是最后一层，不进行剪枝
        }
        
        for layer_name, stats in gradient_stats.items():
            if layer_name == 'classifier.7':  # 跳过最后一层
                continue
                
            # 只处理FC层（跳过IF层，因为我们不直接对IF层剪枝）
            if stats.get('layer_type') != 'fc':
                continue
            
         
            # 根据sort_by选择排序数据
            if sort_by == 'improved_method':
                # 使用改进的方法计算神经元重要性
                if layer_name not in fc_to_if_mapping:
                    print(f"警告: FC层 {layer_name} 没有对应的IF层，跳过")
                    continue
                
                if_layer_name = fc_to_if_mapping[layer_name]
                if if_layer_name not in gradient_stats:
                    print(f"警告: 对应的IF层 {if_layer_name} 没有梯度数据，跳过")
                    continue
                
                if_stats = gradient_stats[if_layer_name]
                if_output_grad = if_stats.get('output_grad_values')
                if_input_grad = if_stats.get('input_grad_values')
                
                if if_output_grad is None:
                    print(f"警告: IF层 {if_layer_name} 没有 output_grad_values 数据，跳过")
                    continue
                
                if if_input_grad is None:
                    print(f"警告: IF层 {if_layer_name} 没有 input_grad_values 数据，跳过")
                    continue
                
                # 获取FC层的权重数据
                fc_weight = None
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        # 计算每个输出神经元的平均权重值
                        fc_weight = module.weight.data.abs().mean(dim=1).cpu().numpy()
                        break
                
                if fc_weight is None:
                    print(f"警告: 找不到FC层 {layer_name} 的权重数据，跳过")
                    continue
                
                # 根据改进的方法计算神经元重要性
                # grad_values = fc_weight / (if_input_grad/if_output_grad)
                grad_values = fc_weight / (if_output_grad/if_input_grad)
                
                # 检查维度匹配
                fc_output_size = None
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        fc_output_size = module.out_features
                        break
                
                if fc_output_size is None:
                    print(f"警告: 找不到FC层 {layer_name}，跳过")
                    continue
                
                if len(grad_values) != fc_output_size:
                    print(f"警告: FC层 {layer_name} 输出维度({fc_output_size}) 与 IF层 {if_layer_name} 梯度维度({len(grad_values)}) 不匹配，跳过")
                    continue
                
                sort_data = grad_values
            else:
                print(f"警告: 未知的排序依据 '{sort_by}'，跳过该层")
                continue
            
            # 根据排序方式对数据进行排序
            if order == 'low' :
                sorted_indices = np.argsort(sort_data)  # 从小到大排序
            elif order == 'high':
                sorted_indices = np.argsort(sort_data)[::-1]  # 从大到小排序
            elif order == 'index':
                sorted_indices = np.arange(len(sort_data))  # 按神经元序号排序
            elif order == 'random':
                # 随机打乱索引
                sorted_indices = np.random.permutation(len(sort_data))
            else:
                # 默认按从小到大排序
                sorted_indices = np.argsort(sort_data)
            
            # 计算要选择的神经元数量
            num_select = int(len(sort_data) * ratio)
            
            # 收集神经元信息
            for idx in sorted_indices[:num_select]:
                # 获取权重信息
                weight_info = 0.0
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_data = module.weight.data
                        weight_info = weight_data.abs().mean(dim=1).cpu().numpy()[idx]
                        break
                
                # 构造神经元信息
                neuron_info = {
                    'layer': layer_name,
                    'neuron_index': int(idx),
                    'sort_value': float(sort_data[idx]),  # 用于排序的值
                    'sort_by': sort_by,  # 排序依据
                }
                
                # 如果使用improved_method，记录额外的信息
                # if sort_by == 'improved_method':
                #     neuron_info['gradient_type'] = 'improved_method'
                #     neuron_info['if_layer'] = fc_to_if_mapping[layer_name]
                #     neuron_info['if_gradient_value'] = float(sort_data[idx])
                
                neurons_to_prune.append(neuron_info)
        
        print(f"get_selected_neurons: 总共选择了 {len(neurons_to_prune)} 个神经元进行剪枝")
        return neurons_to_prune

    def get_high_gradient_neurons(self, gradient_stats, ratio=0.1, sort_by='gradient'):
        """选择高梯度/高权重/权重*梯度的前 ratio 比例神经元。
        improved_method: 基于改进的神经元重要性评估方法进行神经元选取
        """
        high_neurons = []
        
        # FC层与IF层的对应关系映射（用于improved_method）
        fc_to_if_mapping = {
            'classifier.1': 'classifier.2',  # FC层 -> 对应的IF层
            'classifier.4': 'classifier.5',  # FC层 -> 对应的IF层
            # classifier.6 是最后一层，不进行剪枝
        }
        
        for layer_name, stats in gradient_stats.items():
            if layer_name == 'classifier.7':
                continue
            if 'values' not in stats or stats['values'] is None:
                continue
            grads = np.array(stats['values'])

            # 选择排序依据
            if sort_by == 'gradient':
                sort_data = grads
            elif sort_by == 'weight':
                sort_data = grads
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        sort_data = module.weight.data.abs().mean(dim=1).cpu().numpy()
                        break
            elif sort_by == 'weight_gradient':
                sort_data = grads
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weights = module.weight.data.abs().mean(dim=1).cpu().numpy()
                        sort_data = weights * grads
                        break
            elif sort_by == 'improved_method':
                # 使用改进的方法计算神经元重要性
                if layer_name not in fc_to_if_mapping:
                    print(f"警告: FC层 {layer_name} 没有对应的IF层，跳过")
                    continue
                
                if_layer_name = fc_to_if_mapping[layer_name]
                if if_layer_name not in gradient_stats:
                    print(f"警告: 对应的IF层 {if_layer_name} 没有梯度数据，跳过")
                    continue
                
                if_stats = gradient_stats[if_layer_name]
                if_output_grad = if_stats.get('output_grad_values')
                if_input_grad = if_stats.get('input_grad_values')
                
                if if_output_grad is None:
                    print(f"警告: IF层 {if_layer_name} 没有 output_grad_values 数据，跳过")
                    continue
                
                if if_input_grad is None:
                    print(f"警告: IF层 {if_layer_name} 没有 input_grad_values 数据，跳过")
                    continue
                
                # 获取FC层的权重数据
                fc_weight = None
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        # 计算每个输出神经元的平均权重值
                        fc_weight = module.weight.data.abs().mean(dim=1).cpu().numpy()
                        break
                
                if fc_weight is None:
                    print(f"警告: 找不到FC层 {layer_name} 的权重数据，跳过")
                    continue
                
                # 根据改进的方法计算神经元重要性
                # grad_values = fc_weight / (if_input_grad/if_output_grad)
                grad_values = fc_weight / (if_output_grad/if_input_grad)
                
                # 检查维度匹配
                fc_output_size = None
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        fc_output_size = module.out_features
                        break
                
                if fc_output_size is None:
                    print(f"警告: 找不到FC层 {layer_name}，跳过")
                    continue
                
                if len(grad_values) != fc_output_size:
                    print(f"警告: FC层 {layer_name} 输出维度({fc_output_size}) 与 IF层 {if_layer_name} 梯度维度({len(grad_values)}) 不匹配，跳过")
                    continue
                
                sort_data = grad_values
            else:
                sort_data = grads

            sorted_indices = np.argsort(sort_data)[::-1]
            num_select = max(1, int(len(sort_data) * ratio))
            for idx in sorted_indices[:num_select]:
                # 计算权重信息
                weight_info = 0.0
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_info = module.weight.data.abs().mean(dim=1).cpu().numpy()[idx]
                        break
                high_neurons.append({
                    'layer': layer_name,
                    'neuron_index': int(idx),
                    'grad_value': float(grads[idx]),
                    'weight_value': float(weight_info),
                    'weight_gradient_value': float(weight_info * grads[idx]),
                    'sort_value': float(sort_data[idx]),
                    'sort_by': sort_by,
                })
        return high_neurons

    def print_gradient_analysis(self, gradient_stats):
        """打印梯度分析结果"""
        # print("="*80)
        # print("全连接层梯度分布分析")
        # print("="*80)
        
        if not gradient_stats:
            print("没有收集到梯度数据")
            return
        
        for layer_name, stats in gradient_stats.items():
            if 'values' not in stats or stats['values'] is None:
                continue
                
            # print(f"\n层: {layer_name}")
            # print(f"  神经元数量: {stats['num_neurons']:,}")
            # print(f"  梯度统计:")
            # print(f"    均值: {stats['mean']:.8f}")
            # print(f"    标准差: {stats['std']:.8f}")
            # print(f"    最小值: {stats['min']:.8f}")
            # print(f"    最大值: {stats['max']:.8f}")
            # print(f"    中位数: {stats['median']:.8f}")
            # print(f"  梯度分布:")
            # print(f"    25%分位数: {stats['p25']:.8f}")
            # print(f"    75%分位数: {stats['p75']:.8f}")
            # print(f"    95%分位数: {stats['p95']:.8f}")
            # print("-"*60)
        
        # 分析低梯度神经元
        # print("\n低梯度神经元分析:")
        for ratio in [0.05, 0.1, 0.2]:
            low_neurons = self.get_low_gradient_neurons(gradient_stats,'low', ratio)
            # print(f"  梯度最低 {ratio*100:.1f}% 的神经元数量: {len(low_neurons)}")
            
            if low_neurons:
                # 按层分组统计
                layer_counts = {}
                for neuron in low_neurons:
                    layer = neuron['layer']
                    if layer not in layer_counts:
                        layer_counts[layer] = 0
                    layer_counts[layer] += 1
                
                for layer, count in layer_counts.items():
                    # print(f"    {layer}: {count} 个")
                    pass
        
        # print("="*80)
    
    def prune_neurons(self, neurons_to_prune):
        """执行神经元剪枝"""
        # 统计每层剪枝的神经元数量
        layer_prune_count = {}
        
        for neuron_info in neurons_to_prune:
            layer_name = neuron_info['layer']
            neuron_idx = neuron_info['neuron_index']
            
            # 更新统计信息
            if layer_name not in layer_prune_count:
                layer_prune_count[layer_name] = 0
            layer_prune_count[layer_name] += 1
            
            # 找到对应的层
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name and isinstance(mod, nn.Linear):
                    module = mod
                    break
            
            if module:
                # 执行剪枝：将神经元的权重置零
                with torch.no_grad():
                    module.weight.data[neuron_idx] = 0
                    if module.bias is not None:
                        module.bias.data[neuron_idx] = 0
        
        # 打印每层剪枝统计信息
        # print("\n剪枝统计信息:")
        # print("="*60)
        # print(f"{'层名称':<30} {'剪枝神经元数量':<15} {'总神经元数量':<15}")
        # print("-"*60)
        
        total_pruned = 0
        for layer_name, count in layer_prune_count.items():
            # 获取该层的总神经元数量
            for name, module in self.model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    total_neurons = module.out_features
                    # print(f"{layer_name:<30} {count:<15} {total_neurons:<15}")
                    total_pruned += count
                    break
        
        # print("-"*60)
        # print(f"总计剪枝神经元数量: {total_pruned}")
        # print("="*60)

    def snapshot_before_pruning(self):
        """保存剪枝前模型参数的快照,用于后续神经元再生时恢复。"""
        self.before_pruning_state = {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def get_original_weight(self, layer_name, neuron_idx):
        """从快照中获取某层某个神经元对应的原始权重行。"""
        if not hasattr(self, 'before_pruning_state'):
            return None
        key = f"{layer_name}.weight"
        if key not in self.before_pruning_state:
            return None
        weight_tensor = self.before_pruning_state[key]
        if neuron_idx < 0 or neuron_idx >= weight_tensor.shape[0]:
            return None
        return weight_tensor[neuron_idx].detach().clone()

    def get_original_bias(self, layer_name, neuron_idx):
        """从快照中获取某层某个神经元对应的原始偏置值。"""
        if not hasattr(self, 'before_pruning_state'):
            return None
        key = f"{layer_name}.bias"
        if key not in self.before_pruning_state:
            return None
        bias_tensor = self.before_pruning_state[key]
        if neuron_idx < 0 or neuron_idx >= bias_tensor.shape[0]:
            return None
        return bias_tensor[neuron_idx].detach().clone()

    def regenerate_neurons(self, neurons_to_regenerate):
        """再生神经元: 从剪枝前快照恢复被剪枝神经元的权重与偏置。"""
        for neuron_info in neurons_to_regenerate:
            layer_name = neuron_info['layer']
            neuron_idx = neuron_info['neuron_index']

            # 找到对应的层
            for name, mod in self.model.named_modules():
                if name == layer_name and isinstance(mod, nn.Linear):
                    module = mod
                    break

            with torch.no_grad():
                orig_w_row = self.get_original_weight(layer_name, neuron_idx)
                if orig_w_row is not None and neuron_idx < module.weight.data.shape[0]:
                    module.weight.data[neuron_idx] = orig_w_row.to(module.weight.data.device)
                if module.bias is not None:
                    orig_b = self.get_original_bias(layer_name, neuron_idx)
                    if orig_b is not None and neuron_idx < module.bias.data.shape[0]:
                        module.bias.data[neuron_idx] = orig_b.to(module.bias.data.device)
        
    def PruningAndRegeneration(self, gradient_stats, regeneration_ratio=0.1, prune_ratio=0.5, prune_ratio1=0.5, prune_ratio2=0.5):
        """剪枝后再生,引入随机剪枝与定向剪枝结合"""
        
        # 在剪枝开始前保存参数快照
        self.snapshot_before_pruning()
        prune_neurons = self.get_low_gradient_neurons(gradient_stats, 'random', prune_ratio1)

        # 随机选择一部分神经元进行再生
        #regeneration_ratio = (prune_ratio2 - prune_ratio) / prune_ratio1
        regeneration_ratio = 0.0
        num_to_regenerate = int(len(prune_neurons) * regeneration_ratio)
        neurons_to_regenerate = random.sample(prune_neurons, num_to_regenerate) if num_to_regenerate > 0 else []

        # 随机剪枝&定向剪枝
        low_gradient_neurons = self.get_low_gradient_neurons(gradient_stats, 'low', prune_ratio2, 'weight')
        self.prune_neurons(prune_neurons)
        self.prune_neurons(low_gradient_neurons)

        # 执行再生
        self.regenerate_neurons(neurons_to_regenerate)


    def adjust_neurons(self, gradient_stats, adjust_method, lr, device, check_interval, max_steps=1000, threshold_ratio_high=0.6, threshold_ratio_low=0.6, scale_factor_high=1.2, scale_factor_low=1.2, data_loader=None):
        """寻找高梯度神经元"""
        high_gradient_neurons = self.get_high_gradient_neurons(gradient_stats, ratio=0.1, sort_by='gradient')
        #high_gradient_neurons = self.get_high_gradient_neurons(gradient_stats, ratio=0.1, sort_by='improved_method')

        print(f"找到 {len(high_gradient_neurons)} 个高梯度神经元")
        #for neuron in high_gradient_neurons:
            #print(f"  - 层 {neuron['layer']} 神经元 {neuron['neuron_index']}: 梯度值 {neuron['grad_value']:.6f}")
        
        # 打印模型的所有层名称，帮助调试
        #print("\n模型层名称:")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, IF):
                #print(f"  {name}: {type(module).__name__}")
                pass
        
        device = next(self.model.parameters()).device
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # 在统计阶段使用评估模式，避免 BatchNorm 等层的运行统计发生变化
        original_training = self.model.training
        self.model.eval()
        # 2. 重置所有IF层的critical_count
        for module in self.model.modules():
            if isinstance(module, IF):
                if hasattr(module, 'critical_count') and module.critical_count is not None:
                    module.critical_count.zero_()

        # 3. 运行小批量数据并定期检查
        step = 0
        for inputs, targets in data_loader:
            if step >= max_steps:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                self.model(inputs)  # 前向传播，IF层内部会更新critical_count
            
            step += 1

            # 4. 每隔check_interval步检查并调整阈值
            if step % check_interval == 0:
                #print(f"\n--- 步数 {step}: 检查并调整阈值 ---")
                #print(f"检查 {len(high_gradient_neurons)} 个高梯度神经元")
                for neuron_info in high_gradient_neurons:
                    layer_name = neuron_info['layer']
                    neuron_idx = neuron_info['neuron_index']

                    # FC -> IF 映射
                    fc_to_if_mapping = {
                        'classifier.1': 'classifier.2',
                        'classifier.4': 'classifier.5',
                    }
                    if_layer_name = fc_to_if_mapping.get(layer_name, None)
                    if if_layer_name is None:
                        print(f"  跳过层 {layer_name}: 未找到对应的IF层映射")
                        continue
                    if_layer = dict(self.model.named_modules()).get(if_layer_name)
                    if if_layer is None:
                        print(f"  跳过层 {layer_name}: 未找到IF层 {if_layer_name}")
                        continue
                    #print(f"  检查层 {layer_name} -> {if_layer_name}, 神经元 {neuron_idx}")
                    T = getattr(if_layer, 'T', 0)

                    # 获取 critical_count
                    if hasattr(if_layer, 'critical_count') and if_layer.critical_count is not None:
                        # 检查索引是否在范围内
                        if if_layer.critical_count.numel() <= neuron_idx:
                            print(f"    跳过神经元 {neuron_idx}: critical_count大小({if_layer.critical_count.numel()}) <= 神经元索引({neuron_idx})")
                            continue
                        
                        count = float(if_layer.critical_count[neuron_idx].item())
                        
                        total_checks = check_interval * max(1, T)
                        ratio = count / total_checks if total_checks > 0 else 0
                        
                        #print(f"  检查神经元 {neuron_idx}: count={count:.1f}, total_checks={total_checks}, ratio={ratio:.3f}")

                        if ratio > threshold_ratio_high or ratio < threshold_ratio_low:
                            with torch.no_grad():
                                if adjust_method == 'threshold':
                                    
                                    # 检查索引是否在范围内
                                    if if_layer.thresh.numel() > neuron_idx:
                                        original_thresh = float(if_layer.thresh[neuron_idx].item())
                                        if ratio > threshold_ratio_high:
                                            if_layer.thresh[neuron_idx] *= scale_factor_high
                                            used_threshold = threshold_ratio_high
                                        else:
                                            if_layer.thresh[neuron_idx] *= scale_factor_low
                                            used_threshold = threshold_ratio_low
                                        new_thresh = float(if_layer.thresh[neuron_idx].item())
                                        #print(f"  - 层 {layer_name} 神经元 {neuron_idx}: critical_count比例 {ratio:.3f} 触发阈值 {used_threshold}，阈值从 {original_thresh:.4f} 调整为 {new_thresh:.4f}")
                                    else:
                                        used_threshold = threshold_ratio_high if ratio > threshold_ratio_high else threshold_ratio_low
                                        #print(f"  - 层 {layer_name} 神经元 {neuron_idx}: critical_count比例 {ratio:.3f} 触发阈值 {used_threshold}，但阈值索引超出范围，跳过调整")
                                        
                                elif adjust_method == 'weight':
                                    target_module = dict(self.model.named_modules()).get(layer_name)
                                    if target_module is None or not isinstance(target_module, nn.Linear):
                                        used_threshold = threshold_ratio_high if ratio > threshold_ratio_high else threshold_ratio_low
                                        #print(f"  - 层 {layer_name} 神经元 {neuron_idx}: critical_count比例 {ratio:.3f} 触发阈值 {used_threshold}，但未找到对应的Linear层，跳过权重调整")
                                        continue
                                    
                                    # 检查权重索引是否在范围内
                                    if target_module.weight.shape[0] > neuron_idx:
                                        original_norm = float(torch.norm(target_module.weight.data[neuron_idx, :]).item())
                                        scale_factor_used = scale_factor_high if ratio > threshold_ratio_high else scale_factor_low
                                        target_module.weight.data[neuron_idx, :] *= scale_factor_used
                                        new_norm = float(torch.norm(target_module.weight.data[neuron_idx, :]).item())
                                        used_threshold = threshold_ratio_high if ratio > threshold_ratio_high else threshold_ratio_low
                                        #print(f"  - 层 {layer_name} 神经元 {neuron_idx}: critical_count比例 {ratio:.3f} 触发阈值 {used_threshold}，输入权重范数从 {original_norm:.4f} 调整为 {new_norm:.4f}")

                                        # 进行模型权重微调
                                        self.model.train()
                                        optimizer.zero_grad()
                                        outputs = self.model(inputs)
                                        if len(outputs.shape) > 2:
                                            outputs = outputs.mean(0)
                                        loss = criterion(outputs, targets)
                                        loss.backward()
                                        optimizer.step()
                                        print(f"  -> 微调完成, Loss: {loss.item():.4f}")
                                        # 微调后恢复为评估模式，继续统计阶段的前向
                                        self.model.eval()
                                    else:
                                        used_threshold = threshold_ratio_high if ratio > threshold_ratio_high else threshold_ratio_low
                                        #print(f"  - 层 {layer_name} 神经元 {neuron_idx}: critical_count比例 {ratio:.3f} 触发阈值 {used_threshold}，但权重索引超出范围，跳过调整")
                                        
            # 在每次检查后重置 critical_count，开始新的计数周期
            if step % check_interval == 0:
                for module in self.model.modules():
                    if isinstance(module, IF):
                        if hasattr(module, 'critical_count') and module.critical_count is not None:
                            module.critical_count.zero_()

        # 恢复进入本函数前的训练/评估状态
        if original_training:
            self.model.train()
        else:
            self.model.eval()
        
        
    def cleanup_hooks(self):
        """清理梯度钩子"""
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records = {}


    def save_neuronidx_weight_grad(self, model, gradient_stats, timestamp, before_pruning_state=None):
        """
        保存权重分析信息到CSV文件
        
        参数:
        model - 模型
        gradient_stats - 梯度统计信息
        timestamp - 时间戳
        before_pruning_state - 剪枝前的模型状态
        """
        # print("\n开始保存权重分析信息...")
        
        # 确保log目录存在
        log_dir = "log_weight_grad"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"创建目录: {log_dir}")
        
        # 遍历所有全连接层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # print(f"分析层: {name}")
                
                # 获取剪枝后的权重数据
                weight_data = module.weight.data
                out_features, in_features = weight_data.shape
                
                # 计算剪枝后的平均权重
                pruned_weights = weight_data.abs().mean(dim=1).cpu().numpy()
                
                # 获取剪枝前的平均权重
                if before_pruning_state is not None and f"{name}.weight" in before_pruning_state:
                    before_weight_data = before_pruning_state[f"{name}.weight"]
                    avg_befor_weights = before_weight_data.abs().mean(dim=1).cpu().numpy()
                else:
                    # 如果没有剪枝前状态，抛出异常
                    raise ValueError(f"层 {name} 没有剪枝前状态信息，无法保存权重分析数据")
                
                # 获取对应的梯度值
                if name not in gradient_stats or gradient_stats[name]['values'] is None:
                    raise ValueError(f"层 {name} 没有梯度信息，无法保存权重分析数据")
                
                gradient_values = gradient_stats[name]['values']
                
                # 创建DataFrame
                df = pd.DataFrame({
                    'neuron_index': range(out_features),
                    'avg_befor_weight': avg_befor_weights,
                    'pruned_weight': pruned_weights,
                    'gradient_value': gradient_values
                })
                
                # 生成文件名（简化格式）
                filename = f"{name}_weight_grad_{timestamp}.csv"
                filepath = os.path.join(log_dir, filename)
                
                # 保存到CSV
                df.to_csv(filepath, index=False)
        #         print(f"  已保存权重信息到: {filepath}")
        #         print(f"  神经元数量: {out_features}")
        #         print(f"  剪枝前权重统计:")
        #         print(f"    均值: {np.mean(avg_befor_weights):.8f}")
        #         print(f"    标准差: {np.std(avg_befor_weights):.8f}")
        #         print(f"    最小值: {np.min(avg_befor_weights):.8f}")
        #         print(f"    最大值: {np.max(avg_befor_weights):.8f}")
        #         print(f"    中位数: {np.median(avg_befor_weights):.8f}")
        #         print(f"  剪枝后权重统计:")
        #         print(f"    均值: {np.mean(pruned_weights):.8f}")
        #         print(f"    标准差: {np.std(pruned_weights):.8f}")
        #         print(f"    最小值: {np.min(pruned_weights):.8f}")
        #         print(f"    最大值: {np.max(pruned_weights):.8f}")
        #         print(f"    中位数: {np.median(pruned_weights):.8f}")
        #         print(f"  梯度值统计:")
        #         print(f"    均值: {np.mean(gradient_values):.8f}")
        #         print(f"    标准差: {np.std(gradient_values):.8f}")
        #         print(f"    最小值: {np.min(gradient_values):.8f}")
        #         print(f"    最大值: {np.max(gradient_values):.8f}")
        #         print(f"    中位数: {np.median(gradient_values):.8f}")
        
        # print("权重分析信息保存完成!")

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

def evaluate_model(model, test_loader, criterion, device, seed=42):
    """评估模型性能"""
    # 设置随机种子，确保数据加载顺序一致
    seed_all(seed)
    
    # 保存模型原始状态
    original_state = {
        'training': model.training,
        'state_dict': model.state_dict().copy()
    }
    
    model.eval()  # 确保模型在评估模式
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 处理SNN输出
            if len(outputs.shape) > 2:
                outputs = outputs.mean(0)  # 对时间维度求平均
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            
            # 累加统计信息
            total_correct += correct
            total_samples += total
            total_loss += loss.item()
    
    # 计算平均准确率和损失
    avg_accuracy = 100 * total_correct / total_samples
    avg_loss = total_loss / len(test_loader)
    # print(f"\n评估完成:")
    # print(f"  总样本数: {total_samples}")
    print(f"  平均准确率: {avg_accuracy:.2f}% ({total_correct}/{total_samples})")
    print(f"  平均损失: {avg_loss:.6f}")
    
    # 恢复模型原始状态（critical_count 不再是 buffer，不会影响状态恢复）
    model.load_state_dict(original_state['state_dict'])
    
    if original_state['training']:
        model.train()
    else:
        model.eval()
    
    return avg_accuracy, avg_loss

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取VGG16参数和梯度')
    parser.add_argument('--batch_size', default=200, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--num_batches', default=5, type=int, help='梯度分析的批次数')
    parser.add_argument('-r','--pruning_ratio', default=0.5, type=float, help='剪枝比例')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', help='数据集')
    parser.add_argument('--order', default='low', type=str, help='low/high/index/random 从小到大/从大到小/按神经元序号/随机排序')
    parser.add_argument('--sort_by', default='weight', type=str, choices=['gradient', 'weight', 'weight_gradient'], 
                       help='排序依据: gradient(按梯度), weight(按权重), weight_gradient(按梯度*权重)')
    
    args = parser.parse_args()
    
    # 设置输出重定向（默认保存到文件）
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"gradient_analysis_{args.mode}_{timestamp}.txt"
    # output_redirector = OutputRedirector(filename)
    # sys.stdout = output_redirector
    # print(f"输出将保存到文件: {filename}")
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    # print(f"设备: {device}, 随机种子: {args.seed}")
    # print(f"分析模式: {args.mode}")
    # print(f"梯度分析批次数: {args.num_batches}")
    # print(f"剪枝比例: {args.pruning_ratio}")
    # print(f"数据集: {args.dataset}")
    # print(f"排序方式: {args.order}")
    # print(f"排序依据: {args.sort_by}")
    
    # 创建模型
    model = modelpool('vgg16', args.dataset)
    
    # 直接加载预训练模型
    model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
    # model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar100-checkpoints/vgg16_L[4].pth'
    
    # print(f"加载预训练模型: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # 处理阈值形状不匹配的问题
    # 预训练模型中的thresh是标量[1]，现在需要扩展为向量[num_neurons]
    for name, param in state_dict.items():
        if name.endswith('.thresh') and param.shape == torch.Size([1]):
            # 找到对应的IF层
            layer_name = name.replace('.thresh', '')
            for module_name, module in model.named_modules():
                if module_name == layer_name and hasattr(module, 'thresh'):
                    # 获取当前模型的阈值形状
                    target_shape = module.thresh.shape
                    if target_shape != torch.Size([1]):
                        # 将标量阈值扩展为向量阈值
                        scalar_value = param.item()
                        vector_thresh = torch.full(target_shape, scalar_value, dtype=param.dtype)
                        state_dict[name] = vector_thresh
                        print(f"扩展阈值: {name} 从 {param.shape} 到 {vector_thresh.shape}")
                    break

    model.load_state_dict(state_dict, strict=False)
    # 使用兼容性加载函数
    # try:
    #     load_model_compatible(model, state_dict)
    # except Exception as e:
    #     print(f"兼容性加载失败，尝试常规加载: {e}")
    #     model.load_state_dict(state_dict, strict=False)  # 使用非严格模式作为备选

    # print("✅ 预训练模型加载成功")
    
    if args.mode == 'snn':
        model.set_T(8)
        model.set_L(4)
        # print("设置为SNN模式")
    else:
        model.set_T(0)
        model.set_L(4)
        # print("设置为ANN模式")
    
    model.to(device)
    
    # 加载数据
    # print(f"加载{args.dataset}测试数据集...")
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    
    # 使用测试集进行评估
    criterion = nn.CrossEntropyLoss()
    
    # 保存模型初始状态（深拷贝）
    initial_state = {}
    for key, value in model.state_dict().items():
        initial_state[key] = value.clone().detach()
    
    # 剪枝前评估
    print("\n剪枝前评估:")
    pre_accuracy, pre_loss = evaluate_model(model, test_loader, criterion, device, args.seed)
    
    # 分析全连接层梯度
    # print("\n" + "="*80)
    # print("开始全连接层梯度分析")
    # print("="*80)
    
    # 创建梯度分析器
    analyzer = GradientAnalyzer(model)
    
    try:
        # 分析梯度分布
        gradient_stats = analyzer.analyze_gradients(
            train_loader, 
            criterion, 
            num_batches=args.num_batches
        )    
        
        # 剪枝后评估
        # 做一次神经元调整
        
        try:
            analyzer.adjust_neurons(
                gradient_stats=gradient_stats,
                #gradient_stats=gradient_stats_improved,
                adjust_method='threshold',  # 或 'weight'
                lr=1e-4,
                device=device,
                check_interval=5,          # 每5步检查一次
                max_steps=50,              # 少量步数预热
                threshold_ratio_high=0.16,  
                threshold_ratio_low=0.11,
                scale_factor_high=1.1  ,    # 增加调整幅度
                scale_factor_low=1.3,       # 增加调整幅度
                data_loader=train_loader
            )
        except Exception as e:
            print(f"调整神经元时出现问题，跳过调整: {e}")
        
        
        
        # 保存调整后的模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        threshold_ratio = 0.16  # 阈值调整比例
        adjusted_model_path = f'/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/adjusted_model_thresh_{threshold_ratio}_{timestamp}.pth'
        torch.save(model.state_dict(), adjusted_model_path)
        print(f"\n✅ 已保存调整后的模型到: {adjusted_model_path}")
        print(f"   阈值调整比例: {threshold_ratio}")
        #print(f"   调整后准确率: {post_accuracy:.2f}%")


        gradient_stats_improved = analyzer.analyze_gradients_improved(
            train_loader, 
            criterion, 
            num_batches=args.num_batches
        )

        # analyzer.print_gradient_analysis(gradient_stats)    
        
        # 获取低梯度神经元
        # low_gradient_neurons = analyzer.get_low_gradient_neurons(
        #     gradient_stats, 
        #     order=args.order,
        #     ratio=args.pruning_ratio,
        #     sort_by=args.sort_by
        # )
        low_gradient_neurons = analyzer.get_selected_neurons(
            gradient_stats_improved, 
            order=args.order,
            ratio=args.pruning_ratio,
            sort_by="improved_method"
        )    

        
        # 执行剪枝,
        analyzer.prune_neurons(low_gradient_neurons)
        #analyzer.PruningAndRegeneration(gradient_stats, regeneration_ratio=0.0, prune_ratio=0.9, prune_ratio1=0.7, prune_ratio2=0.7)
        print("\n剪枝后评估:")
        post_accuracy, post_loss = evaluate_model(model, test_loader, criterion, device, args.seed)

        # 保存各层梯度信息到CSV文件
        
        for layer_name, stats in gradient_stats.items():
            if 'values' not in stats or stats['values'] is None:
                continue
            
            # 获取该层的神经元数量
            num_neurons = stats['num_neurons']
            
            # 创建DataFrame
            df = pd.DataFrame({
                'neuron_index': range(num_neurons),
                'gradient_value': stats['values']
            })
            
            # 生成文件名
            # filename = f"gradient_analysis_{layer_name}_{timestamp}.csv"
            
            # 保存到CSV
            # df.to_csv(filename, index=False)
            # print(f"已保存{layer_name}层的梯度信息到: {filename}")
            # print(f"  神经元数量: {num_neurons}")
            # print(f"  每个神经元包含1个平均梯度值")
                  
    finally:
        # 保存权重分析信息
        # analyzer.save_neuronidx_weight_grad(model, gradient_stats, timestamp, initial_state)
        
        # 清理梯度钩子
        analyzer.cleanup_hooks()
    
    # print("\n✅ 完成!")

if __name__ == "__main__":
    main() 