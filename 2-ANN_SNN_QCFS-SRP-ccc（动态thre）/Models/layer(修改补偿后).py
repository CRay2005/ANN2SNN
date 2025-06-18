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
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, layer_name="IF"):
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
        self.layer_name = layer_name
    
    

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

            # 正式处理
            mem = 0.50 * thre  # 初始化膜电位            
            # mem = torch.zeros_like(x[0, ...])           

            for t in range(self.T):
                mem = mem + x[t, ...]
                # =======================================================
                # spike = self.act(mem - thre, self.gama) * thre
                spike = self.act(mem - thre, self.gama) 
                # 处理dead neuron
                # spike = spike * deadneuron_flag  
                mem = mem - spike* thre

                spike_pot.append(spike)
                # =======================================================

                if t == self.T-1 :

                    compen_mem = (mem - thre/2)
                    
                    # 1.计算mem对应的spike的数量
                    spike_count = torch.stack(spike_pot, dim=0).sum(dim=0)  # 计算每个位置在所有时间步的spike总数
                    
                    # 2.如果（compen_mem + spike的数量*thre）>0,
                    # 则new_thre=（compen_mem + spike的数量*thre）/（spike的数量），
                    # 否则，new_thre=0
                    # 增加判断：如果(compen_mem + spike_count * thre) > self.T*thre，则取值为self.T*thre
                    compensated_value = compen_mem + spike_count * thre
                    compensated_value = torch.where(
                        compensated_value > self.T * thre,
                        self.T * thre,
                        compensated_value
                    )
                    
                    condition = compensated_value > 0
                    spike_count_safe = torch.where(spike_count > 0, spike_count, torch.ones_like(spike_count))  # 避免除零
                    new_thre = torch.where(
                        condition & (spike_count > 0),
                        compensated_value / spike_count_safe,
                        torch.zeros_like(compen_mem)
                    )
                    # 3.将new_thre乘以spike_pot中的元素
                    for i in range(len(spike_pot)):
                        spike_pot[i] = spike_pot[i] * new_thre
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)

        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh

            # x = x / self.thresh
            # x = myfloor(x*self.L+0.5)/self.L
            # x = torch.clamp(x, 0, 1)            
            # x = x * self.thresh
        return x

class StructuredPruningIF(nn.Module):
    """结构化剪枝的IF层 - 按通道整体剪枝"""
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, 
                 channel_pruning_ratio=0.0):
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
        
        # 结构化剪枝参数
        self.channel_pruning_ratio = channel_pruning_ratio
        self.channel_mask = None
        
    def compute_channel_importance(self, x):
        """计算通道重要性分数"""
        # 基于L2范数和激活频率的综合评估
        l2_norm = torch.norm(x.view(x.size(0), x.size(1), -1), dim=2, p=2)
        activation_freq = (x > 0).float().mean(dim=[0, 2, 3])
        importance_score = 0.7 * l2_norm.mean(0) + 0.3 * activation_freq
        return importance_score
        
    def update_channel_mask(self, importance_scores):
        """更新通道剪枝掩码"""
        if self.channel_pruning_ratio > 0:
            num_channels = importance_scores.size(0)
            num_pruned = int(num_channels * self.channel_pruning_ratio)
            
            if num_pruned > 0:
                _, indices = torch.topk(importance_scores, num_pruned, largest=False)
                self.channel_mask = torch.ones_like(importance_scores)
                self.channel_mask[indices] = 0
        
    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            
            # 计算通道重要性并更新掩码
            importance_scores = self.compute_channel_importance(x)
            self.update_channel_mask(importance_scores)
            
            mem = 0.5 * thre
            spike_pot = []
            
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                
                # 应用通道剪枝
                if self.channel_mask is not None:
                    mask = self.channel_mask.view(1, -1, 1, 1)
                    spike = spike * mask
                    
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
