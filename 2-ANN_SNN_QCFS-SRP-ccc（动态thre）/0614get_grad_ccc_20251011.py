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
from utils import mem_distribution
import pandas as pd
import warnings
from Models.layer import IF
# from Models.layer import load_model_compatible

# 设置环境变量抑制cuDNN警告
os.environ['CUDNN_V8_API_DISABLED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
# 抑制PyTorch相关警告
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class ComprehensiveNeuronAnalyzer:
    """神经元梯度分析器"""
    def __init__(self, model):
        self.model = model
        self.weight_grad_hooks = {}
        self.tensor_grad_hooks = {}
        self.if_grad_hooks = {}
        self.gradient_records = {}
        
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
        self.gradient_records = {}
        
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
                self.gradient_records[name] = {
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
                self.gradient_records[name] = {
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
                self.gradient_records[name]['weight_grad'] = neuron_weight_grads.detach().cpu()
        return hook
    
    def _tensor_gradient_hook(self, name):
        """张量梯度钩子"""
        def hook(module, grad_input, grad_output):
            # 捕获输入梯度
            if grad_input[0] is not None:
                input_grad = grad_input[0]  # [batch_size, in_features]
                # 计算每个输入神经元的平均梯度
                neuron_input_grads = input_grad.abs().mean(dim=0)  # [in_features]
                self.gradient_records[name]['input_grad'] = neuron_input_grads.detach().cpu()
            
            # 捕获输出梯度
            if grad_output[0] is not None:
                output_grad = grad_output[0]  # [batch_size, out_features]
                # 计算每个输出神经元的平均梯度
                neuron_output_grads = output_grad.abs().mean(dim=0)  # [out_features]
                self.gradient_records[name]['output_grad'] = neuron_output_grads.detach().cpu()
        return hook
    
    def _threshold_gradient_hook(self, name):
        """IF层阈值梯度钩子"""
        def hook(grad):
            if grad is not None:
                # 阈值梯度是标量
                thresh_grad = grad.abs().item()
                self.gradient_records[name]['threshold_grad'] = thresh_grad
        return hook
    
    def _if_tensor_gradient_hook(self, name):
        """IF层张量梯度钩子"""
        def hook(module, grad_input, grad_output):
            # 保存阈值数值
            self.gradient_records[name]['threshold_value'] = module.thresh.data.item()
            
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
                self.gradient_records[name]['input_grad'] = neuron_input_grads.detach().cpu()
            
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
                self.gradient_records[name]['output_grad'] = neuron_output_grads.detach().cpu()
        return hook
    

    
    def analyze_gradients(self, dataloader, criterion, num_batches=5):
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
        
        # 注册梯度钩子
        self.register_comprehensive_hooks()
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 梯度统计收集器
        gradient_stats = {}
        for name in self.gradient_records.keys():
            layer_type = self.gradient_records[name]['layer_type']
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
            for name, records in self.gradient_records.items():
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
        
        return gradient_stats

    def analyze_gradient_correlation(self, gradient_stats):
        """分析不同梯度类型之间的相关性"""
        print("\n" + "="*80)
        print("梯度相关性分析")
        print("="*80)
        
        try:
            import scipy.stats
        except ImportError:
            print("需要安装scipy库: pip install scipy")
            return
        
        # FC层与IF层的对应关系映射
        fc_to_if_mapping = {
            'classifier.1': 'classifier.2',
            'classifier.4': 'classifier.5',
        }
        
        for layer_name, stats in gradient_stats.items():
            # 只分析FC层
            if stats.get('layer_type') != 'fc':
                continue
                
            print(f"\n层: {layer_name}")
            
            weight_grad = stats.get('weight_grad_values')
            output_grad = stats.get('output_grad_values') 
            input_grad = stats.get('input_grad_values')
            
            # 获取对应IF层的梯度（如果存在）
            if_output_grad = None
            if layer_name in fc_to_if_mapping:
                if_layer_name = fc_to_if_mapping[layer_name]
                if if_layer_name in gradient_stats:
                    if_stats = gradient_stats[if_layer_name]
                    if_output_grad = if_stats.get('output_grad_values')
            
            # 分析FC层内部梯度的相关性
            if weight_grad is not None and output_grad is not None:
                # 计算权重梯度和输出梯度的相关系数
                corr_coef, p_value = scipy.stats.pearsonr(weight_grad, output_grad)
                print(f"  权重梯度 vs 输出梯度:")
                print(f"    皮尔逊相关系数: {corr_coef:.6f}")
                print(f"    P值: {p_value:.2e}")
                
                # 计算排序相关性（这对剪枝更重要）
                from scipy.stats import spearmanr
                rank_corr, rank_p = spearmanr(weight_grad, output_grad)
                print(f"    斯皮尔曼等级相关系数: {rank_corr:.6f}")
                print(f"    P值: {rank_p:.2e}")
                
                # 分析梯度比值的分布
                if np.all(output_grad > 1e-10):  # 避免除零
                    ratio = weight_grad / output_grad
                    print(f"    权重梯度/输出梯度 比值统计:")
                    print(f"      均值: {ratio.mean():.6f}")
                    print(f"      标准差: {ratio.std():.6f}")
                    print(f"      变异系数: {ratio.std()/ratio.mean():.6f}")
            
            # 分析FC层与对应IF层梯度的相关性
            if if_output_grad is not None:
                if_layer_name = fc_to_if_mapping[layer_name]
                print(f"\n  FC层 vs 对应IF层({if_layer_name})梯度相关性:")
                
                # FC层权重梯度 vs IF层输出梯度
                if weight_grad is not None:
                    if len(weight_grad) == len(if_output_grad):
                        corr_coef, p_value = scipy.stats.pearsonr(weight_grad, if_output_grad)
                        rank_corr, rank_p = spearmanr(weight_grad, if_output_grad)
                        print(f"    FC权重梯度 vs IF输出梯度:")
                        print(f"      皮尔逊相关系数: {corr_coef:.6f} (P值: {p_value:.2e})")
                        print(f"      斯皮尔曼相关系数: {rank_corr:.6f} (P值: {rank_p:.2e})")
                
                # FC层输出梯度 vs IF层输出梯度
                if output_grad is not None:
                    if len(output_grad) == len(if_output_grad):
                        corr_coef, p_value = scipy.stats.pearsonr(output_grad, if_output_grad)
                        rank_corr, rank_p = spearmanr(output_grad, if_output_grad)
                        print(f"    FC输出梯度 vs IF输出梯度:")
                        print(f"      皮尔逊相关系数: {corr_coef:.6f} (P值: {p_value:.2e})")
                        print(f"      斯皮尔曼相关系数: {rank_corr:.6f} (P值: {rank_p:.2e})")
            
            # 分析剪枝神经元的重叠度
            if weight_grad is not None and output_grad is not None:
                print(f"\n  FC层内部梯度剪枝重叠度:")
                # 计算最低神经元的重叠度
                for ratio in [0.1, 0.2, 0.3]:
                    num_prune = int(len(weight_grad) * ratio)
                    
                    weight_indices = set(np.argsort(weight_grad)[:num_prune])
                    output_indices = set(np.argsort(output_grad)[:num_prune])
                    
                    overlap = len(weight_indices & output_indices)
                    overlap_ratio = overlap / num_prune if num_prune > 0 else 0
                    
                    print(f"    最低{ratio*100:.0f}%神经元重叠度: {overlap}/{num_prune} ({overlap_ratio:.2%})")
            
            # 分析FC层梯度与IF层梯度的剪枝重叠度
            if if_output_grad is not None and weight_grad is not None:
                if len(weight_grad) == len(if_output_grad):
                    print(f"\n  FC层 vs IF层剪枝重叠度:")
                    for ratio in [0.1, 0.2, 0.3]:
                        num_prune = int(len(weight_grad) * ratio)
                        
                        fc_weight_indices = set(np.argsort(weight_grad)[:num_prune])
                        if_output_indices = set(np.argsort(if_output_grad)[:num_prune])
                        
                        overlap = len(fc_weight_indices & if_output_indices)
                        overlap_ratio = overlap / num_prune if num_prune > 0 else 0
                        
                        print(f"    FC权重梯度 vs IF输出梯度 最低{ratio*100:.0f}%重叠度: {overlap}/{num_prune} ({overlap_ratio:.2%})")
        
        print("="*80)

    def get_comprehensive_pruning_neurons(self, gradient_stats, ratio=0.1, method='weight_grad_values'):
        """
        基于梯度重要性进行剪枝
        
        参数:
        gradient_stats - analyze_gradients返回的统计数据
        ratio - 要剪枝的神经元比例
        method - 梯度类型: 'weight_grad_values', 'input_grad_values', 'output_grad_values', 'IF_output_grad_values'
        
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
            
            # 根据method选择梯度类型
            if method == 'IF_output_grad_values':
                # 使用对应IF层的output梯度
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
                
                # 获取IF层前一个FC层的权重和权重梯度
                fc_weight_grad = stats.get('weight_grad_values')
                if fc_weight_grad is None:
                    print(f"警告: FC层 {layer_name} 没有 weight_grad_values 数据，跳过")
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
                
                # 计算四个值的加和：IF层输出梯度 + IF层输入梯度 + FC层权重 + FC层权重梯度
                # grad_values = if_output_grad + if_input_grad + fc_weight + fc_weight_grad
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
                
            else:
                # 使用FC层自身的梯度
                grad_values = stats.get(method)
                if grad_values is None:
                    print(f"警告: 层 {layer_name} 没有 {method} 数据，跳过")
                    continue
            
            # 转换为张量进行排序
            grad_tensor = torch.tensor(grad_values, dtype=torch.float32)
            
            # 排序并选择要剪枝的神经元（梯度值最小的）
            sorted_indices = torch.argsort(grad_tensor)
            num_prune = int(len(grad_tensor) * ratio)
            
            for idx in sorted_indices[:num_prune]:
                # 获取权重信息
                weight_info = 0.0
                for name, module in self.model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        weight_data = module.weight.data
                        weight_info = weight_data.abs().mean(dim=1).cpu().numpy()[idx.item()]
                        break
                
                # 构造神经元信息
                neuron_info = {
                    'layer': layer_name,
                    'neuron_index': idx.item(),
                    'gradient_type': method,
                    'gradient_value': grad_values[idx.item()],
                    'weight_value': weight_info
                }
                
                # 如果使用IF层梯度，记录对应的IF层信息
                if method == 'IF_output_grad_values':
                    neuron_info['if_layer'] = fc_to_if_mapping[layer_name]
                    neuron_info['if_gradient_value'] = grad_values[idx.item()]
                
                neurons_to_prune.append(neuron_info)
        
        return neurons_to_prune

    def print_gradient_analysis(self, gradient_stats):
        """打印梯度分析结果"""
        print("="*80)
        print("梯度分析结果")
        print("="*80)
        
        if not gradient_stats:
            print("没有收集到梯度数据")
            return
        
        for layer_name, stats in gradient_stats.items():
            print(f"\n层: {layer_name}")
            
            # 打印权重梯度统计
            if 'weight_grad_values' in stats and stats['weight_grad_values'] is not None:
                weight_grad = np.array(stats['weight_grad_values'])
                print(f"  权重梯度统计:")
                print(f"    均值: {weight_grad.mean():.8f}")
                print(f"    标准差: {weight_grad.std():.8f}")
                print(f"    最小值: {weight_grad.min():.8f}")
                print(f"    最大值: {weight_grad.max():.8f}")
                print(f"    神经元数量: {len(weight_grad)}")
            
            # 打印输入梯度统计
            if 'input_grad_values' in stats and stats['input_grad_values'] is not None:
                input_grad = np.array(stats['input_grad_values'])
                print(f"  输入梯度统计:")
                print(f"    均值: {input_grad.mean():.8f}")
                print(f"    标准差: {input_grad.std():.8f}")
                print(f"    最小值: {input_grad.min():.8f}")
                print(f"    最大值: {input_grad.max():.8f}")
                print(f"    神经元数量: {len(input_grad)}")
            
            # 打印输出梯度统计
            if 'output_grad_values' in stats and stats['output_grad_values'] is not None:
                output_grad = np.array(stats['output_grad_values'])
                print(f"  输出梯度统计:")
                print(f"    均值: {output_grad.mean():.8f}")
                print(f"    标准差: {output_grad.std():.8f}")
                print(f"    最小值: {output_grad.min():.8f}")
                print(f"    最大值: {output_grad.max():.8f}")
                print(f"    神经元数量: {len(output_grad)}")
            
            print("-"*60)
        
        # 分析低梯度神经元
        print("\n低梯度神经元分析:")
        for method in ['weight_grad_values', 'input_grad_values', 'output_grad_values', 'IF_output_grad_values']:
            print(f"\n基于 {method}:")
            for ratio in [0.05, 0.1, 0.2]:
                low_neurons = self.get_comprehensive_pruning_neurons(gradient_stats, ratio, method)
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
                        if method == 'IF_output_grad_values':
                            # 对于IF梯度方法，显示对应的IF层信息
                            if layer == 'classifier.1':
                                if_layer = 'classifier.2'
                            elif layer == 'classifier.4':
                                if_layer = 'classifier.5'
                            else:
                                if_layer = '无对应IF层'
                            print(f"    {layer} (基于 {if_layer}): {count} 个")
                        else:
                            print(f"    {layer}: {count} 个")
        
        print("="*80)
    
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
        print("\n剪枝统计信息:")
        print("="*60)
        print(f"{'层名称':<30} {'剪枝神经元数量':<15} {'总神经元数量':<15}")
        print("-"*60)
        
        total_pruned = 0
        for layer_name, count in layer_prune_count.items():
            # 获取该层的总神经元数量
            for name, module in self.model.named_modules():
                if name == layer_name and isinstance(module, nn.Linear):
                    total_neurons = module.out_features
                    print(f"{layer_name:<30} {count:<15} {total_neurons:<15}")
                    total_pruned += count
                    break
        
        print("-"*60)
        print(f"总计剪枝神经元数量: {total_pruned}")
        print("="*60)
    
    def cleanup_hooks(self):
        """清理梯度钩子"""
        for handle in self.weight_grad_hooks.values():
            handle.remove()
        for handle in self.tensor_grad_hooks.values():
            handle.remove()
        for handle in self.if_grad_hooks.values():
            handle.remove()
        self.weight_grad_hooks = {}
        self.tensor_grad_hooks = {}
        self.if_grad_hooks = {}
        self.gradient_records = {}

    def save_comprehensive_analysis(self, model, gradient_stats, timestamp, before_pruning_state=None):
        """
        保存梯度信息到CSV文件
        
        参数:
        model - 模型
        gradient_stats - 梯度统计信息
        timestamp - 时间戳
        before_pruning_state - 剪枝前的模型状态（保留参数但不使用）
        """
        print("\n开始保存梯度信息...")
        
        # 确保log目录存在
        log_dir = "log_comprehensive_analysis"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"创建目录: {log_dir}")
        
        # 遍历所有全连接层和IF层
        for name, module in model.named_modules():
            # 处理FC层
            if isinstance(module, nn.Linear):
                # print(f"保存FC层 {name} 的梯度信息...")
                
                # 检查该层是否有梯度数据
                if name not in gradient_stats:
                    print(f"  警告: FC层 {name} 没有梯度数据，跳过")
                    continue
                
                # 获取该层的梯度数据
                layer_stats = gradient_stats[name]
                
                # 分别保存权重梯度和输出梯度（它们的维度相同，都是输出神经元数量）
                weight_grad = layer_stats.get('weight_grad_values')
                output_grad = layer_stats.get('output_grad_values')
                input_grad = layer_stats.get('input_grad_values')
                
                # 保存输出相关梯度（权重梯度 + 输出梯度）
                if weight_grad is not None or output_grad is not None:
                    out_features = module.out_features
                    df_data = {'neuron_index': range(out_features)}
                    
                    if weight_grad is not None:
                        df_data['weight_grad_values'] = weight_grad
                    if output_grad is not None:
                        df_data['output_grad_values'] = output_grad
                    
                    df_output = pd.DataFrame(df_data)
                    filename_output = f"{name}_output_gradients_{timestamp}.csv"
                    filepath_output = os.path.join(log_dir, filename_output)
                    df_output.to_csv(filepath_output, index=False)
                    # print(f"  已保存输出梯度信息到: {filepath_output}")
                    # print(f"  输出神经元数量: {out_features}")
                
                # 保存输入梯度（单独保存，因为维度可能不同）
                if input_grad is not None:
                    in_features = module.in_features
                    df_input_data = {
                        'neuron_index': range(in_features),
                        'input_grad_values': input_grad
                    }
                    
                    df_input = pd.DataFrame(df_input_data)
                    filename_input = f"{name}_input_gradients_{timestamp}.csv"
                    filepath_input = os.path.join(log_dir, filename_input)
                    df_input.to_csv(filepath_input, index=False)
                    # print(f"  已保存输入梯度信息到: {filepath_input}")
                    # print(f"  输入神经元数量: {in_features}")
                
                if weight_grad is None and output_grad is None and input_grad is None:
                    print(f"  警告: FC层 {name} 没有有效的梯度数据")
            
            # 处理IF层
            elif isinstance(module, IF) and name in ['classifier.2', 'classifier.5']:
                # print(f"保存IF层 {name} 的梯度信息...")
                
                # 检查该层是否有梯度数据
                if name not in gradient_stats:
                    print(f"  警告: IF层 {name} 没有梯度数据，跳过")
                    continue
                
                # 获取该层的梯度数据
                layer_stats = gradient_stats[name]
                
                # 保存IF层特有的数据
                threshold_grad = layer_stats.get('threshold_grad_values')
                threshold_val = layer_stats.get('threshold_values')
                input_grad = layer_stats.get('input_grad_values')
                output_grad = layer_stats.get('output_grad_values')
                
                # 保存阈值相关信息
                if threshold_grad is not None or threshold_val is not None:
                    # 创建阈值信息列表
                    thresh_data = []
                    
                    if threshold_grad is not None:
                        thresh_data.append({
                            'data_type': 'threshold_grad',
                            'value': threshold_grad
                        })
                    
                    if threshold_val is not None:
                        thresh_data.append({
                            'data_type': 'threshold_value', 
                            'value': threshold_val
                        })
                    
                    df_thresh = pd.DataFrame(thresh_data)
                    filename_thresh = f"{name}_threshold_info_{timestamp}.csv"
                    filepath_thresh = os.path.join(log_dir, filename_thresh)
                    df_thresh.to_csv(filepath_thresh, index=False)
                    # print(f"  已保存阈值信息到: {filepath_thresh}")
                
                # 保存神经元梯度信息（输入和输出梯度）
                if input_grad is not None or output_grad is not None:
                    neuron_features = len(input_grad) if input_grad is not None else len(output_grad)
                    neuron_data = {'neuron_index': range(neuron_features)}
                    
                    if input_grad is not None:
                        neuron_data['input_grad_values'] = input_grad
                    if output_grad is not None:
                        neuron_data['output_grad_values'] = output_grad
                    
                    df_neurons = pd.DataFrame(neuron_data)
                    filename_neurons = f"{name}_neuron_gradients_{timestamp}.csv"
                    filepath_neurons = os.path.join(log_dir, filename_neurons)
                    df_neurons.to_csv(filepath_neurons, index=False)
                    # print(f"  已保存神经元梯度信息到: {filepath_neurons}")
                    # print(f"  神经元数量: {neuron_features}")
                
                if threshold_grad is None and threshold_val is None and input_grad is None and output_grad is None:
                    print(f"  警告: IF层 {name} 没有有效的梯度数据")
        
        print("FC层和IF层梯度信息保存完成!")

    def save_fc_if_correlation_analysis(self, model, gradient_stats, timestamp, neurons_to_prune=None):
        """
        保存FC层与对应IF层的关联分析数据到CSV文件
        
        参数:
        model - 模型
        gradient_stats - 梯度统计信息
        timestamp - 时间戳
        neurons_to_prune - 要剪枝的神经元列表，用于标识剪枝状态
        """
        print("\n开始保存FC层与IF层关联分析数据...")
        
        # 确保log目录存在
        log_dir = "log_comprehensive_analysis"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"创建目录: {log_dir}")
        
        # FC层与IF层的对应关系映射
        fc_to_if_mapping = {
            'classifier.1': 'classifier.2',
            'classifier.4': 'classifier.5',
        }
        
        # 遍历所有FC层
        for fc_layer_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
                
            # 检查是否有对应的IF层
            if fc_layer_name not in fc_to_if_mapping:
                print(f"  FC层 {fc_layer_name} 没有对应的IF层，跳过")
                continue
                
            if_layer_name = fc_to_if_mapping[fc_layer_name]
            
            # 检查FC层和IF层是否都有梯度数据
            if fc_layer_name not in gradient_stats:
                print(f"  警告: FC层 {fc_layer_name} 没有梯度数据，跳过")
                continue
                
            if if_layer_name not in gradient_stats:
                print(f"  警告: IF层 {if_layer_name} 没有梯度数据，跳过")
                continue
            
            print(f"  处理FC层 {fc_layer_name} 与IF层 {if_layer_name} 的关联数据...")
            
            # 获取FC层数据
            fc_stats = gradient_stats[fc_layer_name]
            fc_weight_grad = fc_stats.get('weight_grad_values')
            
            # 获取FC层权重数据
            fc_weight = None
            for name, mod in model.named_modules():
                if name == fc_layer_name and isinstance(mod, nn.Linear):
                    # 计算每个输出神经元的平均权重值
                    fc_weight = mod.weight.data.abs().mean(dim=1).cpu().numpy()
                    break
            
            if fc_weight is None:
                print(f"  警告: 找不到FC层 {fc_layer_name} 的权重数据，跳过")
                continue
            
            # 获取IF层数据
            if_stats = gradient_stats[if_layer_name]
            if_output_grad = if_stats.get('output_grad_values')
            if_input_grad = if_stats.get('input_grad_values')
            
            if if_output_grad is None or if_input_grad is None:
                print(f"  警告: IF层 {if_layer_name} 缺少梯度数据，跳过")
                continue
            
            # 检查维度匹配
            fc_output_size = module.out_features
            if len(fc_weight) != fc_output_size:
                print(f"  警告: FC层 {fc_layer_name} 权重维度({len(fc_weight)}) 与输出维度({fc_output_size}) 不匹配，跳过")
                continue
                
            if len(if_output_grad) != fc_output_size or len(if_input_grad) != fc_output_size:
                print(f"  警告: FC层 {fc_layer_name} 输出维度({fc_output_size}) 与 IF层 {if_layer_name} 梯度维度不匹配，跳过")
                continue
            
            # 计算剪枝依据值
            # try:
            #     # 避免除零错误
            #     if np.any(if_output_grad < 1e-10):
            #         print(f"  警告: IF层 {if_layer_name} 输出梯度包含接近零的值，跳过")
            #         continue
                    
            #     if np.any(if_input_grad < 1e-10):
            #         print(f"  警告: IF层 {if_layer_name} 输入梯度包含接近零的值，跳过")
            #         continue
                
            #     # 计算剪枝依据值: fc_weight / (if_output_grad/if_input_grad)
            #     pruning_criteria = fc_weight / (if_output_grad / if_input_grad)
                
            # except Exception as e:
            #     print(f"  警告: 计算剪枝依据值时出错: {e}，跳过")
            #     continue
            
            # 创建剪枝标识列表
            pruning_status = [1] * fc_output_size  # 默认所有神经元都保留（1表示保留）
            if neurons_to_prune is not None:
                # 遍历要剪枝的神经元，将对应位置标记为0（剪枝）
                for neuron_info in neurons_to_prune:
                    if neuron_info['layer'] == fc_layer_name:
                        neuron_idx = neuron_info['neuron_index']
                        if 0 <= neuron_idx < fc_output_size:
                            pruning_status[neuron_idx] = 0  # 0表示剪枝
            
            # 创建DataFrame
            df_data = {
                'neuron_id': range(fc_output_size),
                'fc_layer_name': [fc_layer_name] * fc_output_size,
                'if_layer_name': [if_layer_name] * fc_output_size,
                'fc_weight': fc_weight,
                'fc_weight_grad': fc_weight_grad if fc_weight_grad is not None else [0.0] * fc_output_size,
                'if_output_grad': if_output_grad,
                'if_input_grad': if_input_grad,
                'if_grad_ratio': if_output_grad / if_input_grad,  # 输出梯度/输入梯度
                'pruning_status': pruning_status,  # 剪枝状态：1=保留，0=剪枝
               # 'pruning_criteria': pruning_criteria,  # fc_weight / (if_output_grad/if_input_grad)
            }
            
            df = pd.DataFrame(df_data)
            
            # 保存到CSV文件
            filename = f"fc_if_correlation_{fc_layer_name}_{if_layer_name}_{timestamp}.csv"
            filepath = os.path.join(log_dir, filename)
            df.to_csv(filepath, index=False)
            
            # 统计剪枝信息
            pruned_count = sum(1 for status in pruning_status if status == 0)
            kept_count = sum(1 for status in pruning_status if status == 1)
            
            print(f"    已保存关联分析数据到: {filepath}")
            print(f"    神经元数量: {fc_output_size}")
            print(f"    剪枝状态统计:")
            print(f"      保留神经元: {kept_count} 个 (标识为1)")
            print(f"      剪枝神经元: {pruned_count} 个 (标识为0)")
            # print(f"    剪枝依据值统计:")
            # print(f"      均值: {pruning_criteria.mean():.8f}")
            # print(f"      标准差: {pruning_criteria.std():.8f}")
            # print(f"      最小值: {pruning_criteria.min():.8f}")
            # print(f"      最大值: {pruning_criteria.max():.8f}")
        
        print("FC层与IF层关联分析数据保存完成!")

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
    
    # 恢复模型原始状态
    model.load_state_dict(original_state['state_dict'])
    if original_state['training']:
        model.train()
    else:
        model.eval()
    
    return avg_accuracy, avg_loss

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='神经元梯度分析和剪枝')
    parser.add_argument('--batch_size', default=200, type=int, help='批次大小')
    parser.add_argument('--device', default='0', type=str, help='设备')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--mode', choices=['ann', 'snn'], default='snn', help='模式')
    parser.add_argument('--num_batches', default=5, type=int, help='梯度分析的批次数')
    parser.add_argument('-r','--pruning_ratio', default=0.8, type=float, help='剪枝比例')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100'], default='cifar10', help='数据集')
    parser.add_argument('--gradient_method', default='IF_output_grad_values', type=str, 
                       choices=['weight_grad_values', 'input_grad_values', 'output_grad_values', 'IF_output_grad_values'],
                       help='梯度类型选择（IF_output_grad_values表示使用对应IF层的输出梯度对FC层剪枝）')
    parser.add_argument('--save_analysis', action='store_true', help='是否保存梯度分析结果')
    parser.add_argument('--print_analysis', action='store_true', help='是否打印详细分析结果')
    
    args = parser.parse_args()
    
    # 设置环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    
    print(f"设备: {device}, 随机种子: {args.seed}")
    print(f"分析模式: {args.mode}")
    print(f"梯度分析批次数: {args.num_batches}")
    print(f"剪枝比例: {args.pruning_ratio}")
    print(f"数据集: {args.dataset}")
    print(f"梯度类型: {args.gradient_method}")
    
    # 创建模型
    model = modelpool('vgg16', args.dataset)
    
    # 直接加载预训练模型
    model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar10-checkpoints/vgg16_wd[0.0005].pth'
    # model_path = '/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/cifar100-checkpoints/vgg16_L[4].pth'
    
    print(f"加载预训练模型: {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    print("✅ 预训练模型加载成功")
    
    if args.mode == 'snn':
        model.set_T(4)
        model.set_L(4)
        print("设置为SNN模式: T=8, L=4")
    else:
        model.set_T(0)
        model.set_L(4)
        print("设置为ANN模式: T=0, L=4")
    
    model.to(device)
    
    # 加载数据
    print(f"加载{args.dataset}数据集...")
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
    
    # 创建神经元梯度分析器
    print("\n" + "="*80)
    print("开始神经元梯度分析")
    print("="*80)
    
    analyzer = ComprehensiveNeuronAnalyzer(model)
    
    try:
        # 分析梯度分布
        gradient_stats = analyzer.analyze_gradients(
            train_loader, 
            criterion, 
            num_batches=args.num_batches
        )
        
        # 打印梯度分析结果
        if args.print_analysis:
            analyzer.print_gradient_analysis(gradient_stats)
            analyzer.analyze_gradient_correlation(gradient_stats)

        # 检测神经元膜电位分布
        # targets = {
        #     'classifier.2': [0, 5, 10, 15, 20, 25, 35, 40, 50, 60, 70, 80, 90],
        #     'classifier.5': [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63],
        # }
        # mem_distribution(model, test_loader, device, targets, max_batches=None, bins=50, save_dir='mem_plots', show=False)
        
        # 获取要剪枝的神经元
        neurons_to_prune = analyzer.get_comprehensive_pruning_neurons(
            gradient_stats, 
            ratio=args.pruning_ratio,
            method=args.gradient_method
        )
        
        print(f"\n基于梯度分析，将剪枝 {len(neurons_to_prune)} 个神经元")

        # 保存梯度分析信息（剪枝前）
        if args.save_analysis:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analyzer.save_comprehensive_analysis(model, gradient_stats, timestamp, initial_state)
            # 保存FC层与IF层关联分析数据，包含剪枝信息
            analyzer.save_fc_if_correlation_analysis(model, gradient_stats, timestamp, neurons_to_prune)
               
        # 执行剪枝
        analyzer.prune_neurons(neurons_to_prune)
        
        # 检测神经元膜电位分布
        #targets = {
        #    'classifier.2': [0, 5, 10, 15, 20, 25, 35, 40, 50, 60, 70, 80, 90],
        #    'classifier.5': [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63],
        #}
        #mem_distribution(model, test_loader, device, targets, max_batches=None, bins=50, save_dir='mem_plots', show=False)
        
        # 剪枝后评估
        print("\n剪枝后评估:")
        post_accuracy, post_loss = evaluate_model(model, test_loader, criterion, device, args.seed)

        # # 打印性能对比
        # print("\n性能对比:")
        print(f"剪枝前: 准确率 {pre_accuracy:.2f}%, 损失 {pre_loss:.6f}")
        print(f"剪枝后: 准确率 {post_accuracy:.2f}%, 损失 {post_loss:.6f}")
        print(f"准确率变化: {post_accuracy - pre_accuracy:+.2f}%")
        print(f"损失变化: {post_loss - pre_loss:+.6f}")
        
           
    finally:
        # 清理梯度钩子
        analyzer.cleanup_hooks()
    
    # print("\n✅ 神经元梯度分析和剪枝完成!")
    # print("\n💡 使用说明:")
    # print("python 0614get_grad_ccc.py --mode snn --gradient_method weight_grad_values --print_analysis")
    # print("python 0614get_grad_ccc.py --mode snn --pruning_ratio 0.3 --save_analysis")
    # print("python 0614get_grad_ccc.py --mode ann --num_batches 10 --gradient_method output_grad_values")

if __name__ == "__main__":
    main() 