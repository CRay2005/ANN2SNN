import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm


def group_product(xs, ys):
    """计算两个张量列表的内积"""
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def get_params_grad(model):
    """获取模型参数和对应的梯度"""
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """计算Hessian向量乘积 Hv"""
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, 
                            only_inputs=True, retain_graph=not stop_criterion)
    return hv


class HessianWeightImportance:
    """基于Hutchinson方法的Hessian权重重要性计算器"""
    
    def __init__(self, model, device='cuda', n_samples=300):
        self.model = model
        self.device = device
        self.n_samples = n_samples
        
        # 存储结果
        self.layer_params = []
        self.hessian_traces = {}
        self.weight_importance = {}
        
        # 准备模型参数
        self._prepare_model()
    
    def _prepare_model(self):
        """准备模型参数，过滤不需要的层"""
        print("准备模型参数...")
        
        for name, param in self.model.named_parameters():
            # 过滤掉不需要计算重要性的参数
            if any(skip in name.lower() for skip in ['bias', 'bn', 'batch_norm']):
                continue
            
            if param.requires_grad and len(param.shape) >= 2:  # 只考虑权重矩阵
                self.layer_params.append((name, param))
                print(f"注册参数层: {name} - {param.shape}")
    
    def compute_hessian_trace_hutchinson(self, data_loader, criterion):
        """使用Hutchinson方法计算Hessian trace"""
        print(f"使用Hutchinson方法计算Hessian trace (n_samples={self.n_samples})...")
        
        self.model.eval()
        
        # 获取一个batch用于计算梯度
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx > 0:  # 只用第一个batch
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.model.zero_grad()
            output = self.model(data)
            
            # 处理SNN输出（如果有时间维度）
            if len(output.shape) > 2:
                output = output.mean(0)
            
            # 计算损失和梯度
            loss = criterion(output, target)
            loss.backward(create_graph=True)
            
            # 获取参数和梯度
            params, gradsH = get_params_grad(self.model)
            
            # 为每个参数初始化channel-wise trace存储
            param_traces = {}
            for i, (name, param) in enumerate(self.layer_params):
                if len(param.shape) == 2:  # 全连接层 [out, in]
                    num_channels = param.shape[0]
                elif len(param.shape) == 4:  # 卷积层 [out, in, h, w]
                    num_channels = param.shape[0]
                else:
                    num_channels = param.shape[-1]
                
                param_traces[name] = [[] for _ in range(num_channels)]
            
            # Hutchinson采样
            print("开始Hutchinson采样...")
            progress_bar = tqdm(range(self.n_samples), desc="计算Hessian trace")
            
            for sample_idx in progress_bar:
                # 生成随机向量
                v = [torch.randint_like(p, high=2, device=self.device).float() * 2 - 1 
                     for p in params]
                
                # 计算Hessian-向量乘积
                Hv = hessian_vector_product(gradsH, params, v, 
                                          stop_criterion=(sample_idx == self.n_samples - 1))
                
                # 转移到CPU节省GPU内存
                Hv = [hvi.detach().cpu() for hvi in Hv]
                v = [vi.detach().cpu() for vi in v]
                
                # 计算channel-wise trace
                with torch.no_grad():
                    for i, (name, param) in enumerate(self.layer_params):
                        if i >= len(Hv):
                            continue
                        
                        hv_param = Hv[i]
                        v_param = v[i]
                        
                        # 处理不同维度的参数
                        if len(param.shape) == 2:  # 全连接层 [out, in]
                            for channel_idx in range(hv_param.shape[0]):
                                trace_val = hv_param[channel_idx].flatten().dot(
                                    v_param[channel_idx].flatten()).item()
                                param_traces[name][channel_idx].append(trace_val)
                        
                        elif len(param.shape) == 4:  # 卷积层 [out, in, h, w]
                            for channel_idx in range(hv_param.shape[0]):
                                trace_val = hv_param[channel_idx].flatten().dot(
                                    v_param[channel_idx].flatten()).item()
                                param_traces[name][channel_idx].append(trace_val)
            
            # 计算平均trace
            for name in param_traces:
                channel_traces = []
                for channel_samples in param_traces[name]:
                    if len(channel_samples) > 0:
                        avg_trace = sum(channel_samples) / len(channel_samples)
                        channel_traces.append(avg_trace)
                    else:
                        channel_traces.append(0.0)
                
                self.hessian_traces[name] = channel_traces
                print(f"层 {name}: 计算了 {len(channel_traces)} 个通道的Hessian trace")
            
            break
    
    def compute_weight_importance(self):
        """计算权重重要性: importance = hessian_trace * (weight_norm^2 / num_weights)"""
        print("计算权重重要性...")
        
        for name, param in self.layer_params:
            if name not in self.hessian_traces:
                continue
            
            hessian_trace_list = self.hessian_traces[name]
            importance_list = []
            
            # 处理不同维度的参数
            if len(param.shape) == 2:  # 全连接层 [out, in]
                for channel_idx in range(param.shape[0]):
                    channel_weight = param[channel_idx]  # [in]
                    weight_norm_sq = channel_weight.norm(p=2) ** 2
                    num_weights = channel_weight.numel()
                    hessian_trace = hessian_trace_list[channel_idx]
                    
                    # 核心公式：weight_importance = hessian_trace * (weight_norm^2 / num_weights)
                    importance = hessian_trace * (weight_norm_sq.item() / num_weights)
                    importance_list.append(importance)
            
            elif len(param.shape) == 4:  # 卷积层 [out, in, h, w]
                for channel_idx in range(param.shape[0]):
                    channel_weight = param[channel_idx]  # [in, h, w]
                    weight_norm_sq = channel_weight.norm(p=2) ** 2
                    num_weights = channel_weight.numel()
                    hessian_trace = hessian_trace_list[channel_idx]
                    
                    # 核心公式：weight_importance = hessian_trace * (weight_norm^2 / num_weights)
                    importance = hessian_trace * (weight_norm_sq.item() / num_weights)
                    importance_list.append(importance)
            
            self.weight_importance[name] = importance_list
            
            print(f"层 {name}: 重要性范围 [{min(importance_list):.6f}, {max(importance_list):.6f}]")
    
    def run_full_analysis(self, data_loader, criterion):
        """运行完整的权重重要性分析"""
        print("="*80)
        print("开始基于Hessian的权重重要性分析")
        print("实现公式: weight_importance = hessian_trace * (weight_norm^2 / num_weights)")
        print("="*80)
        
        start_time = time.time()
        
        # 计算Hessian trace
        self.compute_hessian_trace_hutchinson(data_loader, criterion)
        
        # 计算权重重要性
        self.compute_weight_importance()
        
        total_time = time.time() - start_time
        print(f"\n分析完成！总耗时: {total_time:.2f}秒")
        print("="*80)
        
        return {
            'hessian_traces': self.hessian_traces,
            'weight_importance': self.weight_importance
        }