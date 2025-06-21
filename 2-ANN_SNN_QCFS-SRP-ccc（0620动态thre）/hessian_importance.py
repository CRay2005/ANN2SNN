import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from Models.layer import IF


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
    """计算Hessian向量乘积 Hv"""
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, 
                            only_inputs=True, retain_graph=not stop_criterion)
    return hv


class HessianWeightImportance:
    """
    基于HSTNN项目的Hessian权重重要性计算器
    实现公式: weight_importance = hessian_trace * (weight_norm^2 / num_weights)
    """
    
    def __init__(self, model, device='cuda', n_samples=100):
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
        """准备模型参数，只选择IF层的参数"""
        print("准备IF层参数...")
        
        for name, module in self.model.named_modules():
            if isinstance(module, IF):  # 只选择IF层
                for param_name, param in module.named_parameters():
                    if param.requires_grad:
                        full_name = f"{name}.{param_name}"
                        self.layer_params.append((full_name, param))
                        print(f"注册IF层参数: {full_name} - {param.shape}")
    
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
            
            # 只获取我们注册的层的参数和梯度
            params = []
            gradsH = []
            valid_layers = []
            
            for name, param in self.layer_params:
                if param.grad is not None:
                    params.append(param)
                    gradsH.append(param.grad)
                    valid_layers.append((name, param))
                else:
                    print(f"警告: 层 {name} 没有梯度，跳过...")
            
            # 更新有效的层列表
            self.layer_params = valid_layers
            
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
            
            for sample_idx in range(self.n_samples):
                # 生成随机向量 (Rademacher分布: +1 或 -1)
                v = [torch.randint_like(p, high=2, device=self.device).float() * 2 - 1 
                     for p in params]
                
                # 计算Hessian-向量乘积
                Hv = hessian_vector_product(gradsH, params, v, 
                                          stop_criterion=(sample_idx == self.n_samples - 1))
                
                # 转移到CPU节省GPU内存
                Hv = [hvi.detach().cpu() for hvi in Hv]
                v = [vi.detach().cpu() for vi in v]
                
                # 计算channel-wise trace (核心：按通道计算 v^T * H * v)
                with torch.no_grad():
                    for i, (name, param) in enumerate(self.layer_params):
                        if i >= len(Hv):
                            continue
                        
                        hv_param = Hv[i]
                        v_param = v[i]
                        
                        # 按第0维（输出通道）分解计算
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
                
                if (sample_idx + 1) % 10 == 0:
                    print(f"  完成 {sample_idx + 1}/{self.n_samples} 次采样")
            
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
        """计算IF神经元的重要性"""
        print("计算IF神经元重要性...")
        
        for name, param in self.layer_params:
            if name not in self.hessian_traces:
                continue
            
            hessian_trace = self.hessian_traces[name][0]  # IF层只有一个参数
            importance = hessian_trace * (param.norm(p=2) ** 2).item()
            self.weight_importance[name] = [importance]
            
            print(f"IF层 {name}: 重要性 = {importance:.6f}")
    
    def analyze_importance_distribution(self):
        """分析IF神经元重要性分布"""
        print("\nIF神经元重要性分布分析:")
        print("-" * 60)
        
        all_importances = []
        layer_stats = {}
        
        for name, importance_list in self.weight_importance.items():
            all_importances.extend(importance_list)
            
            layer_stats[name] = {
                'importance': importance_list[0],
                'position': name  # 记录IF层在网络中的位置
            }
            
            print(f"{name}:")
            print(f"  重要性: {layer_stats[name]['importance']:.6f}")
        
        print("\n全局统计:")
        print(f"IF层总数: {len(all_importances)}")
        print(f"重要性均值: {np.mean(all_importances):.6f}")
        print(f"重要性标准差: {np.std(all_importances):.6f}")
        print(f"重要性范围: [{np.min(all_importances):.6f}, {np.max(all_importances):.6f}]")
        
        return layer_stats
    
    def get_pruning_candidates(self, pruning_ratio=0.3):
        """获取IF层剪枝候选"""
        print(f"\n生成IF层剪枝候选 (剪枝比例: {pruning_ratio})...")
        
        # 收集所有IF层的重要性分数
        all_importances = []
        for name, importance_list in self.weight_importance.items():
            all_importances.append((name, 0, importance_list[0]))  # 0是占位符，因为IF层只有一个参数
        
        # 按重要性排序
        all_importances.sort(key=lambda x: x[2])
        
        # 选择要剪枝的层
        num_to_prune = int(len(all_importances) * pruning_ratio)
        pruning_candidates = all_importances[:num_to_prune]
        
        print(f"选择了 {len(pruning_candidates)} 个IF层进行剪枝")
        for name, _, importance in pruning_candidates:
            print(f"  {name}: {importance:.6f}")
        
        return pruning_candidates
    
    def run_full_analysis(self, data_loader, criterion):
        """运行完整的权重重要性分析"""
        print("="*80)
        print("基于HSTNN的Hessian权重重要性分析")
        print("实现公式: weight_importance = hessian_trace * (weight_norm^2 / num_weights)")
        print("="*80)
        
        start_time = time.time()
        
        # 1. 计算Hessian trace
        self.compute_hessian_trace_hutchinson(data_loader, criterion)
        
        # 2. 计算权重重要性
        self.compute_weight_importance()
        
        # 3. 分析重要性分布
        layer_stats = self.analyze_importance_distribution()
        
        # 4. 生成剪枝候选
        pruning_candidates = self.get_pruning_candidates(0.3)
        
        total_time = time.time() - start_time
        print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
        print("="*80)
        
        return {
            'hessian_traces': self.hessian_traces,
            'weight_importance': self.weight_importance,
            'layer_stats': layer_stats,
            'pruning_candidates': pruning_candidates
        }