from cv2 import mean
from sympy import print_rcode
from collections import defaultdict
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
            mem = 0.5 * thre  # 初始化膜电位
            spike_pot = []
            
            # # QCFC处理============================================
                        
            # for t in range(self.T):
            #     mem = mem + x[t, ...]
            #     spike = self.act(mem - thre, self.gama) * thre
            #     mem = mem - spike
            #     spike_pot.append(spike)
            # # =======================================================



            # # 模电压补偿处理============================================

            for t in range(self.T):
                mem = mem + x[t, ...]
                # spike = self.act(mem - thre, self.gama) * thre
                spike = self.act(mem - thre, self.gama) 
                mem = mem - spike* thre
                spike_pot.append(spike)

                if t == self.T-1  :
                    compen_mem = (mem - thre/2)                    
                    # 1.计算mem对应的spike的数量
                    spike_count = torch.stack(spike_pot, dim=0).sum(dim=0)  # 计算每个位置在所有时间步的spike总数
                    
                    # 2.如果（compen_mem + spike_count*thre）>0,
                    # 则new_thre=（compen_mem + spike_count*thre）/（spike_count），
                    # 否则，new_thre=0
                    # 增加判断：如果(compen_mem + spike_count * thre) > self.T*thre，则取值为self.T*thre
                    compen_value = compen_mem + spike_count * thre
                    # compen_value = torch.where(
                    #     compen_value > self.T * thre,
                    #     self.T * thre,
                    #     compen_value
                    # )
                    compen_value = torch.clamp(compen_value, max=self.T * thre)

                    # spike_count_safe = torch.where(spike_count > 0, spike_count, torch.ones_like(spike_count))  # 避免除零
                    new_thre = torch.where(
                        (compen_value > 0) & (spike_count > 0),
                        compen_value / spike_count,
                        torch.zeros_like(compen_mem)
                    )
                    
                    # 3.将new_thre乘以spike_pot中的元素
                    for i in range(len(spike_pot)):
                        spike_pot[i] = spike_pot[i] * new_thre
                    
            #         # # 将new_thre信息写入文件
            #         # new_thre_data = new_thre.detach().cpu().numpy()
            #         # with open('/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/hook_outputs/new_thre_data.txt', 'a') as f:
            #         #     f.write(f"Layer:{self.layer_name},Shape:{new_thre_data.shape},Mean:{new_thre_data.mean():.4f},Std:{new_thre_data.std():.4f},Max:{new_thre_data.max():.4f},Min:{new_thre_data.min():.4f},NewThreData:{new_thre_data.flatten()[:10].tolist()}\n")
            #     # =======================================================
                    
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            
            # # 保存最终状态信息到文件
            # mem_data = mem.detach().cpu().numpy()
            # # 统计所有时间步的spike情况
            # total_spike_data = torch.stack(spike_pot, dim=0).detach().cpu().numpy()  # [T, batch, ...]
            # # 将所有时间步的spike累加，统计总的脉冲发放情况
            # spike_sum = total_spike_data.sum(axis=0)  # 对时间维度求和
            # spike_binary = (spike_sum > 0).astype(int)  # 只要发过脉冲就标记为1
            # spike_count = spike_binary.sum()  # 统计发过脉冲的神经元总数
            # # 计算平均发放频率
            # avg_spike_rate = total_spike_data.mean()
            
            # with open('/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/hook_outputs/membrane_data.txt', 'a') as f:
            #     f.write(f"Layer:{self.layer_name},Shape:{mem_data.shape},Threshold:{thre.item():.4f},AvgMem:{mem_data.mean():.4f},MemCount:{mem_data.size},SpikeCount:{spike_count},AvgSpikeRate:{avg_spike_rate:.4f},Spike:{spike_binary.flatten()[:10].tolist()},SpikeSum:{spike_sum.flatten()[:10].tolist()},MemData:{mem_data.flatten()[:10].tolist()}\n")
            
            # # 写入神经元阈值和膜电位到文件
            # with open('/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP/hook_outputs/neuron_data.txt', 'a') as f:
            #     f.write(f"Layer: {self.layer_name}, Threshold: {thre.item():.4f}, Membrane Mean: {mem.mean().item():.4f}\n")
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
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


class GradientAnalyzer:
    def __init__(self, model, prune_ratio=0.1):
        self.model = model
        self.prune_ratio = prune_ratio
        self.gradient_records = defaultdict(list)
        
        # 注册梯度钩子
        for name, param in model.named_parameters():
            if 'weight' in name:
                hook = self.make_hook(name)
                param.register_hook(hook)
    
    def make_hook(self, name):
        def gradient_hook(grad):
            layer_idx = int(name.split('.')[1])  # e.g. 'layers.0.weight'
            grad_mag = grad.abs().mean(dim=1)    # 平均输入梯度
            self.gradient_records[layer_idx].append(grad_mag.detach().cpu())
        return gradient_hook
    
    def get_low_grad_neurons(self):
        all_neuron_grads = {}
        
        # 处理每层梯度记录
        for layer_idx, grads in self.gradient_records.items():
            time_avg_grad = torch.stack(grads).mean(dim=0)  # 时间维度平均
            
            # 神经元梯度统计 (公式1)
            neuron_grads = time_avg_grad.mean(dim=0)  # [神经元] <- 批次平均
            
            # 保存神经元索引和梯度值
            for neuron_id, grad_val in enumerate(neuron_grads):
                key = (layer_idx, neuron_id)
                all_neuron_grads[key] = grad_val.item()
        
        # 筛选梯度最小的神经元 (公式2)
        sorted_neurons = sorted(all_neuron_grads.items(), key=lambda x: x[1])
        num_select = int(len(sorted_neurons) * self.prune_ratio)
        return sorted_neurons[:num_select]  # [(层ID, 神经元ID), ...]



# # 模拟训练步骤
# for data, target in dataloader:
#     optimizer.zero_grad()
#     outputs = model(data, steps=10)
#     loss = loss_fn(outputs, target)
#     loss.backward()
#     optimizer.step()
#     model.reset_states()  # 重置神经元状态

# # 获取低梯度神经元
# low_grad_neurons = analyzer.get_low_grad_neurons()
# print(f"筛选神经元: {len(low_grad_neurons)}/{sum(len(l) for l in model.neurons)}")
# print("前5个低梯度神经元:", low_grad_neurons[:5])

# # 应用神经元筛选结果进行后续处理 (e.g. 剪枝或冻结)
# for (layer_idx, neuron_idx) in low_grad_neurons:
#     prune_neuron(model, layer_idx, neuron_idx)  # 自定义剪枝函数