#!/usr/bin/env python3
"""
改进的Hessian权重重要性计算器
解决数值精度问题和采样不足问题
"""

import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm


def get_params_grad(model):
    """获取模型参数和对应的梯度"""
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        params.append(param)
        grads.append(param.grad)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """计算Hessian向量乘积 Hv，使用高精度"""
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, 
                            only_inputs=True, retain_graph=not stop_criterion)
    return hv


class ImprovedHessianWeightImportance:
    """
    改进的Hessian权重重要性计算器
    专注于全连接层的Hessian分析
    """
    
    def __init__(self, model, device='cuda', n_samples=500, use_double_precision=True):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.use_double_precision = use_double_precision
        
        # 存储结果
        self.layer_params = []
        self.hessian_traces = {}
        self.weight_importance = {}
        
        # 准备模型参数
        self._prepare_model()
    
    def _prepare_model(self):
        """准备全连接层参数"""
        print("准备全连接层参数...")
        
        # 处理全连接层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 保存参数信息
                self.layer_params.append((name, module.weight))
                print(f"注册全连接层: {name} (输入={module.in_features}, 输出={module.out_features})")
    
    def compute_hessian_trace_hutchinson(self, data_loader, criterion):
        """使用改进的Hutchinson方法计算Hessian trace"""
        print(f"使用改进的Hutchinson方法 (n_samples={self.n_samples}, 高精度={self.use_double_precision})...")
        
        # 设置为训练模式
        self.model.train()
        
        # 确保所有参数可训练
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 如果使用双精度，转换模型
        if self.use_double_precision:
            print("转换为双精度模式...")
            self.model = self.model.double()
        
        # 获取多个batch提高稳定性
        all_traces = {}
        for name, _ in self.layer_params:
            all_traces[name] = []
        
        batch_count = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 3:  # 使用前3个batch
                break
                
            batch_count += 1
            data, target = data.to(self.device), target.to(self.device)
            
            if self.use_double_precision:
                data = data.double()
            
            print(f"\n处理第 {batch_idx + 1} 个batch...")
            
            # 前向传播
            self.model.zero_grad()
            output = self.model(data)
            
            # 处理SNN输出
            if len(output.shape) > 2:
                output = output.mean(0)
            
            # 计算损失和梯度
            loss = criterion(output, target)
            loss.backward(create_graph=True)
            
            # 获取参数和梯度
            params = []
            gradsH = []
            valid_layers = []
            
            for name, param in self.layer_params:
                if param.grad is not None:
                    params.append(param)
                    gradsH.append(param.grad)
                    valid_layers.append((name, param))
                    print(f"层 {name} 的梯度范数: {param.grad.norm().item():.8f}")
                else:
                    print(f"警告: 层 {name} 没有梯度")
            
            # Hutchinson采样 - 增加采样数
            batch_traces = {}
            for name, _ in valid_layers:
                batch_traces[name] = []
            
            print(f"开始Hutchinson采样 (批次 {batch_idx + 1})...")
            
            for sample_idx in range(self.n_samples):
                # 使用不同的随机分布提高采样质量
                if sample_idx % 2 == 0:
                    # Rademacher分布
                    v = [torch.randint_like(p, high=2, device=self.device).float() * 2 - 1 
                         for p in params]
                else:
                    # 标准正态分布
                    v = [torch.randn_like(p, device=self.device) for p in params]
                
                if self.use_double_precision:
                    v = [vi.double() for vi in v]
                
                try:
                    # 计算Hessian-向量乘积
                    Hv = hessian_vector_product(gradsH, params, v, 
                                              stop_criterion=(sample_idx == self.n_samples - 1))
                    
                    # 计算trace并保持高精度
                    for i, (name, param) in enumerate(valid_layers):
                        if i >= len(Hv):
                            continue
                        
                        # 直接计算 v^T * H * v
                        trace_val = torch.sum(v[i] * Hv[i]).item()
                        batch_traces[name].append(trace_val)
                    
                    # 清理中间变量
                    del Hv
                    del v
                    
                except Exception as e:
                    print(f"  样本 {sample_idx} 计算失败: {e}")
                    continue
                
                if (sample_idx + 1) % 100 == 0:
                    print(f"  完成 {sample_idx + 1}/{self.n_samples} 次采样")
            
            # 收集这个batch的结果
            for name in batch_traces:
                if len(batch_traces[name]) > 0:
                    avg_trace = np.mean(batch_traces[name])
                    all_traces[name].append(avg_trace)
                    print(f"  批次 {batch_idx + 1} - 层 {name}: 平均trace = {avg_trace:.8f}")
        
        # 计算所有batch的最终平均trace
        print(f"\n计算 {batch_count} 个batch的最终平均...")
        for name in all_traces:
            if len(all_traces[name]) > 0:
                final_trace = np.mean(all_traces[name])
                trace_std = np.std(all_traces[name])
                self.hessian_traces[name] = [final_trace]
                print(f"层 {name}: 最终trace = {final_trace:.8f} ± {trace_std:.8f}")
            else:
                self.hessian_traces[name] = [0.0]
                print(f"层 {name}: 无有效trace，设为0")
    
    def collect_neuron_importance(self):
        """
        收集每个神经元的重要性数据
        
        返回:
        neuron_importance_list - 包含每个神经元重要性信息的列表
        """
        print("\n收集神经元重要性数据...")
        neuron_importance_list = []
        
        for name, param in self.layer_params:
            if name not in self.hessian_traces:
                print(f"警告: {name} 没有Hessian trace")
                continue
            
            # 检查梯度
            if param.grad is None:
                print(f"警告: {name} 没有梯度")
                continue
                
            # 检查Hessian trace
            hessian_trace = self.hessian_traces[name][0]
            if abs(hessian_trace) < 1e-10:
                print(f"警告: {name} 的Hessian trace接近0")
            
            weight_norm_sq = param.norm(p=2) ** 2
            param_count = param.numel()
            
            # 计算每个神经元的重要性
            if param_count > 0:
                base_importance = hessian_trace * weight_norm_sq.item() / param_count
                
                # 对每个神经元计算重要性
                for neuron_idx in range(param.size(0)):  # 遍历输出神经元
                    # 获取该神经元的权重范数
                    neuron_weight_norm = param[neuron_idx].norm(p=2) ** 2
                    # 计算该神经元的重要性
                    neuron_importance = base_importance * neuron_weight_norm.item()
                    
                    # 添加到列表
                    neuron_importance_list.append({
                        'layer': name,
                        'neuron_id': neuron_idx,
                        'importance': neuron_importance
                    })
                    
                    if neuron_idx < 5:  # 只打印前5个神经元的信息作为示例
                        print(f"层 {name} 神经元 {neuron_idx}: 重要性 = {neuron_importance:.8f}")
        
        # 按重要性排序
        neuron_importance_list.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\n总共收集了 {len(neuron_importance_list)} 个神经元的重要性数据")
        return neuron_importance_list
    
    def get_pruning_candidates(self, neuron_importance_list, pruning_ratio=0.3):
        """
        获取剪枝候选，基于神经元重要性排序
        
        参数:
        neuron_importance_list - 神经元重要性列表
        pruning_ratio - 要剪枝的神经元比例
        
        返回:
        pruning_candidates - 剪枝候选列表
        """
        print(f"\n生成剪枝候选 (剪枝比例: {pruning_ratio})...")
        
        # 按重要性排序（从小到大）
        sorted_neurons = sorted(neuron_importance_list, key=lambda x: x['importance'])
        
        # 选择要剪枝的神经元
        num_to_prune = int(len(sorted_neurons) * pruning_ratio)
        pruning_candidates = sorted_neurons[:num_to_prune]
        
        # 按层分组统计
        layer_counts = {}
        for neuron in pruning_candidates:
            layer = neuron['layer']
            if layer not in layer_counts:
                layer_counts[layer] = 0
            layer_counts[layer] += 1
        
        print(f"选择了 {len(pruning_candidates)} 个神经元进行剪枝:")
        for layer, count in layer_counts.items():
            print(f"  {layer}: {count} 个神经元")
        
        return pruning_candidates

    def run_full_analysis(self, data_loader, criterion):
        """运行完整分析"""
        print("="*80)
        print("🚀 全连接层Hessian重要性分析")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 计算Hessian trace
            print("\n1. 计算Hessian trace...")
            self.compute_hessian_trace_hutchinson(data_loader, criterion)
            
            # 收集神经元重要性数据
            print("\n2. 收集神经元重要性数据...")
            neuron_importance_list = self.collect_neuron_importance()
            
            # 生成剪枝候选
            print("\n3. 生成剪枝候选...")
            pruning_candidates = self.get_pruning_candidates(neuron_importance_list)
            
            total_time = time.time() - start_time
            print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
            print("="*80)
            
            return {
                'neuron_importance_list': neuron_importance_list,
                'hessian_traces': self.hessian_traces,
                'pruning_candidates': pruning_candidates
            }
        finally:
            # 清理
            if self.use_double_precision:
                self.model = self.model.float()


# 测试脚本
if __name__ == "__main__":
    print("测试改进的Hessian重要性计算器...")
    
    from Models import modelpool
    from Preprocess import datapool
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = modelpool('vgg16', 'cifar10')
    model.set_L(8)
    model.set_T(0)  # ANN模式
    model.to(device)
    
    # 加载数据
    train_loader, _ = datapool('cifar10', 16)
    
    # 创建改进的计算器
    hessian_calc = ImprovedHessianWeightImportance(
        model=model,
        device=device,
        n_samples=200,  # 增加采样数
        use_double_precision=True  # 使用双精度
    )
    
    # 运行分析
    criterion = nn.CrossEntropyLoss()
    results = hessian_calc.run_full_analysis(train_loader, criterion)
    
    print("\n🎯 分析结果总结:")
    print(f"神经元总数: {len(results['neuron_importance_list'])}")
    if len(results['neuron_importance_list']) > 0:
        print("✅ 成功计算出神经元重要性值！")
    else:
        print("❌ 未能计算出神经元重要性值，需要进一步调试") 