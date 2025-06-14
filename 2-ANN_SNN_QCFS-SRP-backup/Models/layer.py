from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn
import numpy as np

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # print(f"ZIF backward\n")
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    # 类级别的计数器，用于自动分配层ID
    _layer_counter = 0
    
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        
        # 统计相关
        IF._layer_counter += 1
        self.layer_id = IF._layer_counter  # 自动分配层ID
        self.dead_neuron_stats = {}  # 存储死神经元统计信息
        self.forward_count = 0  # 前向传播计数
        
        # Fisher信息矩阵近似（用于权重重要性评估）
        self.fisher_information = None
        self.gradient_accumulator = None

    def compute_dead_neuron_ratio(self, deadneuron_flag):
        """计算死神经元比例"""
        total_neurons = deadneuron_flag.numel()
        dead_neurons = (deadneuron_flag == 0).sum().item()
        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0.0
        return dead_ratio, dead_neurons, total_neurons
    
    def print_dead_neuron_stats(self, dead_ratio, dead_neurons, total_neurons, layer_info=""):
        """打印死神经元统计信息"""
        print(f"DeadNeuron统计{layer_info}: "
              f"死神经元比例={dead_ratio:.4f} ({dead_neurons}/{total_neurons})")
    
    @classmethod
    def reset_layer_counter(cls):
        """重置层计数器（用于新模型）"""
        cls._layer_counter = 0
    
    @classmethod
    def get_layer_count(cls):
        """获取当前层总数"""
        return cls._layer_counter
    
    def get_previous_weight_layer(self):
        """尝试获取前一个权重层（Conv2d或Linear）"""
        # 这需要在模型构建时注册相邻层关系
        # 由于限制只能修改layer.py，这里提供一个概念性实现
        return None
    
    def compute_fisher_based_importance(self, activations):
        """
        计算基于Fisher信息矩阵的神经元重要性
        这是对 hessian_trace * (weight_norm^2 / num_weights) 的近似实现
        
        Fisher信息矩阵近似： F ≈ E[∇log p(y|x) ∇log p(y|x)^T]
        对于神经元重要性，我们计算激活值的统计特性作为近似
        """
        if len(activations.shape) >= 4:  # 卷积层 [B, C, H, W] or [T, B, C, H, W]
            # 计算每个通道的激活方差（近似Fisher信息）
            if len(activations.shape) == 5:  # [T, B, C, H, W]
                # 对时间和空间维度求方差，保留通道维度
                activation_variance = torch.var(activations, dim=[0, 1, 3, 4])
                activation_mean = torch.mean(activations, dim=[0, 1, 3, 4])
            else:  # [B, C, H, W]
                activation_variance = torch.var(activations, dim=[0, 2, 3])
                activation_mean = torch.mean(activations, dim=[0, 2, 3])
            
            # 模拟权重范数的影响：使用激活值的幅度作为权重强度的代理
            weight_proxy = torch.abs(activation_mean) + 1e-8
            
            # 近似 Fisher * weight_norm^2：方差代表敏感性，均值代表权重强度
            importance_score = activation_variance * (weight_proxy ** 2)
            
        elif len(activations.shape) >= 2:  # 全连接层
            if len(activations.shape) == 3:  # [T, B, features]
                activation_variance = torch.var(activations, dim=[0, 1])
                activation_mean = torch.mean(activations, dim=[0, 1])
            else:  # [B, features]
                activation_variance = torch.var(activations, dim=0)
                activation_mean = torch.mean(activations, dim=0)
            
            weight_proxy = torch.abs(activation_mean) + 1e-8
            importance_score = activation_variance * (weight_proxy ** 2)
        else:
            # 默认情况
            importance_score = torch.var(activations) * (torch.mean(torch.abs(activations)) ** 2)
            
        return importance_score
    
    def print_fisher_importance_stats(self, importance_scores, layer_info=""):
        """打印Fisher重要性统计"""
        if importance_scores is not None and self.forward_count <= 2:
            mean_importance = importance_scores.mean().item()
            std_importance = importance_scores.std().item()
            print(f"Fisher重要性{layer_info}: "
                  f"均值={mean_importance:.6f}, 标准差={std_importance:.6f}")
            print(f"重要性分布: min={importance_scores.min().item():.6f}, "
                  f"max={importance_scores.max().item():.6f}")
    
    def update_fisher_information(self, activations):
        """更新Fisher信息矩阵的近似"""
        current_fisher = self.compute_fisher_based_importance(activations)
        
        if self.fisher_information is None:
            self.fisher_information = current_fisher
        else:
            # 指数移动平均更新
            alpha = 0.9
            self.fisher_information = alpha * self.fisher_information + (1 - alpha) * current_fisher
            
        return self.fisher_information

    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            mem = 0.5 * thre
            spike_pot = []
            
            # 预处理===========
            for tp in range(4):
                mem = mem + x[tp, ...]
                spike = self.act(mem - thre, self.gama) * thre
                mem = mem - spike

            deadneuron_flag = torch.where(mem > 1e-3,torch.ones_like(mem),torch.zeros_like(mem))     

            # 统计死神经元比例和Fisher重要性
            self.forward_count += 1
            if self.forward_count == 1 or self.forward_count == 50:  # 在第1次和第100次时打印统计
                dead_ratio, dead_neurons, total_neurons = self.compute_dead_neuron_ratio(deadneuron_flag)
                
                # 计算Fisher重要性（基于预处理后的数据）
                fisher_importance = self.update_fisher_information(x)
                
                # 生成层信息
                layer_info = f"[Layer-{self.layer_id}][Forward-{self.forward_count}]"
                
                self.print_dead_neuron_stats(dead_ratio, dead_neurons, total_neurons, layer_info)
                self.print_fisher_importance_stats(fisher_importance, layer_info)
                
                # 保存统计信息
                self.dead_neuron_stats[f'forward_{self.forward_count}'] = {
                    'dead_ratio': dead_ratio,
                    'dead_neurons': dead_neurons,
                    'total_neurons': total_neurons,
                    'fisher_importance_mean': fisher_importance.mean().item() if fisher_importance is not None else 0,
                    'fisher_importance_std': fisher_importance.std().item() if fisher_importance is not None else 0
                }

            # 正式处理
            mem = 0.5 * thre  # 初始化膜电位            
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                # 处理dead neuron
                spike = spike * deadneuron_flag  
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
        return x


class StructuredPruningIF(nn.Module):
    """改进的IF层 - 支持结构化剪枝 (原StructuredPruningIF)"""
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, 
                 channel_pruning_ratio=0.0, conv_pruning_ratio=0.0, fc_pruning_ratio=0.0):
        super(StructuredPruningIF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        
        # 结构化剪枝参数 - 支持分层剪枝比例
        self.channel_pruning_ratio = channel_pruning_ratio
        self.conv_pruning_ratio = conv_pruning_ratio if conv_pruning_ratio is not None else channel_pruning_ratio
        self.fc_pruning_ratio = fc_pruning_ratio if fc_pruning_ratio is not None else channel_pruning_ratio
        self.channel_mask = None
        self.debug_counter = 0  # 调试计数器
        self.pruning_initialized = False  # 剪枝是否已初始化
        
    def initialize_pruning_mask(self, x):
        """基于第一个batch的数据初始化固定的剪枝掩码"""
        # 根据输入维度确定剪枝比例
        if len(x.shape) >= 4:  # 卷积层
            current_ratio = self.conv_pruning_ratio
            layer_type = "Conv"
        elif len(x.shape) >= 2:  # 全连接层
            current_ratio = self.fc_pruning_ratio
            layer_type = "FC"
        else:
            current_ratio = 0.0
            layer_type = "Unknown"
            
        if self.pruning_initialized or current_ratio <= 0:
            return
            
        print(f"Debug: 初始化{layer_type}层剪枝掩码，ratio={current_ratio}")
        
        # 计算通道重要性（基于第一个batch）
        importance_scores = self.compute_channel_importance(x)
        
        if importance_scores.numel() > 0:
            # 确保importance_scores是1维的
            if importance_scores.dim() > 1:
                importance_scores = importance_scores.flatten()
            
            num_channels = importance_scores.size(0)
            num_pruned = int(num_channels * current_ratio)
            
            print(f"Debug: num_channels={num_channels}, num_pruned={num_pruned}")
            
            if num_pruned > 0 and num_pruned < num_channels:
                # 选择重要性最低的通道进行剪枝
                _, indices = torch.topk(importance_scores, num_pruned, largest=False)
                self.channel_mask = torch.ones_like(importance_scores)
                self.channel_mask[indices] = 0
                
                print(f"Debug: 剪枝初始化完成! 保留{(self.channel_mask.sum()).item()}/{num_channels}个通道")
                print(f"Debug: 被剪枝的通道索引: {indices.tolist()}")
            else:
                # 如果剪枝比例无效，保持所有通道
                self.channel_mask = torch.ones_like(importance_scores)
                print(f"Debug: 剪枝比例无效，保留所有通道")
        
        self.pruning_initialized = True
        
    def compute_channel_importance(self, x):
        """计算通道重要性分数 - 按通道维度计算"""
        # 处理不同维度的输入
        if len(x.shape) >= 4:  # [T, B, C, H, W] 或 [B, C, H, W]
            if len(x.shape) == 5:  # [T, B, C, H, W]
                # 计算每个通道的L2范数：对T, B, H, W维度求范数，保留C维度
                l2_norm = torch.norm(x, dim=[0, 1, 3, 4], p=2)  # 结果: [C]
                # 计算每个通道的激活频率
                activation_freq = (x > 0).float().mean(dim=[0, 1, 3, 4])  # 结果: [C]
            else:  # [B, C, H, W]
                # 计算每个通道的L2范数：对B, H, W维度求范数，保留C维度
                l2_norm = torch.norm(x, dim=[0, 2, 3], p=2)  # 结果: [C]
                # 计算每个通道的激活频率
                activation_freq = (x > 0).float().mean(dim=[0, 2, 3])  # 结果: [C]
            importance_score = 0.7 * l2_norm + 0.3 * activation_freq
        elif len(x.shape) == 3:  # [T, B, features] 全连接层经过expand后
            # 对于全连接层，每个特征就是一个"通道"
            l2_norm = torch.norm(x, dim=[0, 1], p=2)  # 结果: [features]
            activation_freq = (x > 0).float().mean(dim=[0, 1])  # 结果: [features]
            importance_score = 0.7 * l2_norm + 0.3 * activation_freq
        elif len(x.shape) == 2:  # [B, features] 全连接层
            # 对于全连接层，每个特征就是一个"通道"
            l2_norm = torch.norm(x, dim=0, p=2)  # 结果: [features]
            activation_freq = (x > 0).float().mean(dim=0)  # 结果: [features]
            importance_score = 0.7 * l2_norm + 0.3 * activation_freq
        else:
            # 其他情况，使用简单的L2范数
            importance_score = torch.norm(x, dim=0, p=2) if x.size(0) > 1 else torch.norm(x, p=2)
        
        # 调试信息
        if self.debug_counter < 3 and self.channel_pruning_ratio > 0:
            print(f"Debug: x.shape={x.shape}, importance_score.shape={importance_score.shape}")
            print(f"Debug: importance_score min/max/mean = {importance_score.min():.4f}/{importance_score.max():.4f}/{importance_score.mean():.4f}")
            
        return importance_score
        
    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            
            # 只在第一次前向传播时初始化剪枝掩码
            self.initialize_pruning_mask(x)
            
            mem = 0.5 * thre
            spike_pot = []
            
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                
                # 应用固定的通道剪枝掩码
                if self.channel_mask is not None and self.channel_pruning_ratio > 0:
                    # 调试信息
                    if self.debug_counter < 3 and t == 0:
                        print(f"Debug: spike.shape={spike.shape}, channel_mask.shape={self.channel_mask.shape}")
                        spike_mean_before = spike.mean().item()
                        active_channels_before = (spike.abs() > 1e-6).sum().item()
                    
                    # 根据spike的维度调整mask的形状
                    if len(spike.shape) == 4:  # [B, C, H, W]
                        if self.channel_mask.size(0) == spike.size(1):
                            mask = self.channel_mask.view(1, -1, 1, 1)
                            spike = spike * mask
                        else:
                            if self.debug_counter < 3:
                                print(f"Debug: 维度不匹配! channel_mask.size(0)={self.channel_mask.size(0)}, spike.size(1)={spike.size(1)}")
                    elif len(spike.shape) == 2:  # [B, features]
                        if self.channel_mask.size(0) == spike.size(1):
                            spike = spike * self.channel_mask.unsqueeze(0)
                        else:
                            if self.debug_counter < 3:
                                print(f"Debug: 维度不匹配! channel_mask.size(0)={self.channel_mask.size(0)}, spike.size(1)={spike.size(1)}")
                    
                    # 调试信息
                    if self.debug_counter < 3 and t == 0:
                        spike_mean_after = spike.mean().item()
                        active_channels_after = (spike.abs() > 1e-6).sum().item()
                        print(f"Debug: spike均值 剪枝前:{spike_mean_before:.4f} -> 剪枝后:{spike_mean_after:.4f}")
                        print(f"Debug: 活跃元素 剪枝前:{active_channels_before} -> 剪枝后:{active_channels_after}")
                    
                mem = mem - spike
                spike_pot.append(spike)
                
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            
            self.debug_counter += 1  # 增加调试计数器
        else:
            # ANN模式不需要剪枝
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
        return x

class ProgressivePruningIF(nn.Module):
    """渐进式剪枝的IF层 - 逐步增加剪枝力度"""
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0,
                 initial_pruning_ratio=0.0, final_pruning_ratio=0.3, 
                 pruning_schedule_steps=1000):
        super(ProgressivePruningIF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        
        # 渐进式剪枝参数
        self.initial_pruning_ratio = initial_pruning_ratio
        self.final_pruning_ratio = final_pruning_ratio
        self.pruning_schedule_steps = pruning_schedule_steps
        self.current_step = 0
        self.magnitude_scores = None
        self.prune_mask = None
        
    def get_current_pruning_ratio(self):
        """计算当前的剪枝比例 - 使用余弦退火调度"""
        progress = min(self.current_step / self.pruning_schedule_steps, 1.0)
        ratio = self.initial_pruning_ratio + (
            self.final_pruning_ratio - self.initial_pruning_ratio
        ) * (1 - np.cos(np.pi * progress)) / 2
        return ratio
        
    def update_magnitude_scores(self, x):
        """更新权重幅度分数 - 使用指数移动平均"""
        current_scores = torch.norm(x.view(x.size(0), -1), dim=0, p=2)
        
        if self.magnitude_scores is None:
            self.magnitude_scores = current_scores
        else:
            alpha = 0.9
            self.magnitude_scores = alpha * self.magnitude_scores + (1 - alpha) * current_scores
            
    def update_pruning_mask(self):
        """基于当前剪枝比例更新掩码"""
        current_ratio = self.get_current_pruning_ratio()
        
        if current_ratio > 0 and self.magnitude_scores is not None:
            threshold = torch.quantile(self.magnitude_scores, current_ratio)
            self.prune_mask = (self.magnitude_scores > threshold).float()
            
    def forward(self, x):
        self.current_step += 1
        
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            
            self.update_magnitude_scores(x)
            self.update_pruning_mask()
            
            mem = 0.5 * thre
            spike_pot = []
            
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                
                if self.prune_mask is not None:
                    spike_flat = spike.view(spike.size(0), -1)
                    spike_flat = spike_flat * self.prune_mask.unsqueeze(0)
                    spike = spike_flat.view_as(spike)
                    
                mem = mem - spike
                spike_pot.append(spike)
                
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
        return x

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
