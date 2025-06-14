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
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, layer_name="IF", 
                 enable_adaptive_thresh=True, learning_rate=0.1):
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
        
        # 神经元级别的自适应阈值
        self.enable_adaptive_thresh = enable_adaptive_thresh
        self.c = learning_rate  # 学习率
        # 初始化神经元阈值为0，在第一次forward时会初始化为合适的形状
        self.register_buffer("neuron_thre", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))
        
        # 统计信息
        self.register_buffer("update_count", torch.tensor(0))
        self.register_buffer("thresh_diff_sum", torch.tensor(0.0))
        self.register_buffer("thresh_diff_count", torch.tensor(0))
    
    

    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            
            # 初始化神经元级别的阈值
            if not self.initialized:
                self._initialize_neuron_thresh(x)
                self.initialized = torch.tensor(True)
            
            # 选择使用的阈值
            if self.enable_adaptive_thresh:
                # 使用自适应阈值 - 正确处理维度扩展
                if len(x[0].shape) == 4:  # [B, C, H, W] 卷积层
                    current_thre = self.neuron_thre.view(1, -1, 1, 1).expand_as(x[0])
                elif len(x[0].shape) == 2:  # [B, C] 全连接层
                    current_thre = self.neuron_thre.unsqueeze(0).expand_as(x[0])
                else:  # 其他情况使用固定阈值
                    current_thre = thre
            else:
                # 使用固定阈值
                current_thre = thre
            
            mem = 0.5 * current_thre  # 初始化膜电位
            spike_pot = []
            
            # 模电压补偿处理
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - current_thre, self.gama) 
                mem = mem - spike * current_thre
                spike_pot.append(spike)

                if t == self.T-1:
                    compen_mem = (mem - current_thre/2)                    
                    # 1.计算mem对应的spike的数量
                    spike_count = torch.stack(spike_pot, dim=0).sum(dim=0)
                    
                    # 2.计算补偿值
                    compen_value = compen_mem + spike_count * current_thre
                    compen_value = torch.clamp(compen_value, max=self.T * current_thre)

                    # 3.计算new_thre
                    new_thre = torch.where(
                        (compen_value > 0) & (spike_count > 0),
                        compen_value / spike_count,
                        torch.zeros_like(compen_mem)
                    )
                    
                    # 4.更新自适应阈值（仅在训练模式下）
                    if self.enable_adaptive_thresh :
                        self._update_adaptive_thresh(new_thre)
                    
                    # 5.将扩展后的neuron_thre乘以spike_pot中的元素
                    for i in range(len(spike_pot)):
                        # 直接使用自适应阈值，正确扩展维度
                        if len(spike_pot[i].shape) == 4:  # [B, C, H, W] 卷积层
                            thresh_expanded = self.neuron_thre.view(1, -1, 1, 1).expand_as(spike_pot[i])
                        elif len(spike_pot[i].shape) == 2:  # [B, C] 全连接层
                            thresh_expanded = self.neuron_thre.unsqueeze(0).expand_as(spike_pot[i])
                        else:
                            thresh_expanded = self.neuron_thre  # 其他情况直接使用
                        spike_pot[i] = spike_pot[i] * thresh_expanded
                    
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
    
    def _initialize_neuron_thresh(self, x):
        """初始化神经元级别的阈值"""
        # 获取输入的通道数，处理扩展后的时间维度输入 [T, B, C, ...]
        if len(x.shape) == 5:  # [T, B, C, H, W] 卷积层
            num_channels = x.shape[2]
        elif len(x.shape) == 4:  # [T, B, C] 全连接层  
            num_channels = x.shape[2]
        elif len(x.shape) == 3:  # [T, B, neurons]
            num_channels = x.shape[2]
        else:
            num_channels = 1
            
        # 初始化神经元阈值为固定阈值，确保在正确的设备上
        device = x.device
        self.neuron_thre = self.thresh.data.expand(num_channels).clone().to(device)
        self.initialized = torch.tensor(True, device=device)
        
        # print(f"初始化神经元阈值 - 层名称: {self.layer_name}, 通道数: {num_channels}, 输入形状: {x.shape}, 阈值形状: {self.neuron_thre.shape}")
        
    def _update_adaptive_thresh(self, new_thre):
        """使用梯度下降方式更新自适应阈值"""
        if self.neuron_thre.numel() == 1:
            return
            
        # 对所有维度（除了通道维度）求平均，得到每个通道的平均new_thre
        if len(new_thre.shape) == 4:  # [B, C, H, W]
            channel_new_thre = new_thre.mean(dim=[0, 2, 3])  # [C]
        elif len(new_thre.shape) == 2:  # [B, C]  
            channel_new_thre = new_thre.mean(dim=0)  # [C]
        else:
            channel_new_thre = new_thre.mean()
            
        # 确保channel_new_thre与neuron_thre的形状匹配
        if channel_new_thre.numel() != self.neuron_thre.numel():
            # print(f"警告 - 阈值维度不匹配: channel_new_thre {channel_new_thre.shape}, neuron_thre {self.neuron_thre.shape}")
            return
            
        # 参考用户提供的更新公式: self.neuron_thre += self.c * 2 * ((x - ub) * (x > ub).float()).mean()
        ub = channel_new_thre  # 使用通道级别的new_thre作为上界
        diff = ub - self.neuron_thre
        update = self.c * 2 * (diff * (self.neuron_thre > ub).float()).mean()
        
        # 更新神经元阈值
        self.neuron_thre += update
        
        # 记录统计信息 - 确保标量形式一致
        thresh_diff = torch.abs(self.neuron_thre.mean() - ub).detach()
        if thresh_diff.dim() == 0:  # 如果是标量
            thresh_diff = thresh_diff.item()
        else:
            thresh_diff = thresh_diff.mean().item()
            
        self.thresh_diff_sum += thresh_diff
        self.thresh_diff_count += 1
        self.update_count += 1
        
    def get_thresh_stats(self):
        """获取阈值统计信息"""
        if self.thresh_diff_count > 0:
            avg_diff = self.thresh_diff_sum / self.thresh_diff_count
            # 确保avg_diff是Python数值类型
            if hasattr(avg_diff, 'item'):
                avg_diff = avg_diff.item()
            return {
                'neuron_thresh': self.neuron_thre.mean().item() if self.neuron_thre.numel() > 1 else self.neuron_thre.item(),
                'original_thresh': self.thresh.item(),
                'avg_diff': avg_diff,
                'update_count': self.update_count.item()
            }
        else:
            return {
                'neuron_thresh': self.neuron_thre.mean().item() if self.neuron_thre.numel() > 1 else self.neuron_thre.item(),
                'original_thresh': self.thresh.item(),
                'avg_diff': 0.0,
                'update_count': 0
            }
    
    def reset_adaptive_thresh(self):
        """重置自适应阈值到原始值"""
        if self.initialized:
            self.neuron_thre.data.fill_(self.thresh.data.item())
            self.thresh_diff_sum.zero_()
            self.thresh_diff_count.zero_()
            self.update_count.zero_()

    @staticmethod
    def upgrade_state_dict(state_dict, model):
        """升级旧版本模型的state_dict以兼容新的IF层
        
        Args:
            state_dict: 旧版本的模型状态字典
            model: 当前模型实例
            
        Returns:
            upgraded_state_dict: 升级后的状态字典
        """
        upgraded_state_dict = state_dict.copy()
        
        # 为所有IF层添加缺失的buffer参数
        for name, module in model.named_modules():
            if isinstance(module, IF):
                # 添加缺失的neuron_thre buffer
                neuron_thre_key = f"{name}.neuron_thre"
                if neuron_thre_key not in upgraded_state_dict:
                    upgraded_state_dict[neuron_thre_key] = torch.tensor(0.0)
                
                # 添加缺失的initialized buffer
                initialized_key = f"{name}.initialized"
                if initialized_key not in upgraded_state_dict:
                    upgraded_state_dict[initialized_key] = torch.tensor(False)
                
                # 添加缺失的统计信息buffer
                for buffer_name in ['update_count', 'thresh_diff_sum', 'thresh_diff_count']:
                    buffer_key = f"{name}.{buffer_name}"
                    if buffer_key not in upgraded_state_dict:
                        if buffer_name == 'update_count' or buffer_name == 'thresh_diff_count':
                            upgraded_state_dict[buffer_key] = torch.tensor(0)
                        else:  # thresh_diff_sum
                            upgraded_state_dict[buffer_key] = torch.tensor(0.0)
        
        return upgraded_state_dict
    
    def load_compatible_state_dict(self, state_dict, strict=True):
        """兼容性加载状态字典
        
        Args:
            state_dict: 要加载的状态字典
            strict: 是否严格匹配参数名
        """
        # 检查是否存在新增的buffer参数
        missing_keys = []
        for name, module in self.named_modules():
            if isinstance(module, IF):
                for buffer_name in ['neuron_thre', 'initialized', 'update_count', 'thresh_diff_sum', 'thresh_diff_count']:
                    key = f"{name}.{buffer_name}"
                    if key not in state_dict:
                        missing_keys.append(key)
        
        if missing_keys:
            print(f"检测到旧版本模型，自动添加缺失的参数: {len(missing_keys)}个")
            state_dict = IF.upgrade_state_dict(state_dict, self)
        
        return self.load_state_dict(state_dict, strict=strict)

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


def load_model_compatible(model, state_dict, strict=True):
    """
    兼容性模型加载函数，自动处理IF层的新增buffer参数
    
    Args:
        model: 要加载参数的模型
        state_dict: 旧版本的状态字典
        strict: 是否严格匹配参数名
        
    Returns:
        加载结果
    """
    print("开始兼容性模型加载...")
    
    # 检查是否存在新增的buffer参数
    missing_keys = []
    for name, module in model.named_modules():
        if isinstance(module, IF):
            for buffer_name in ['neuron_thre', 'initialized', 'update_count', 'thresh_diff_sum', 'thresh_diff_count']:
                key = f"{name}.{buffer_name}"
                if key not in state_dict:
                    missing_keys.append(key)
    
    if missing_keys:
        print(f"检测到旧版本模型，自动添加缺失的自适应阈值参数: {len(missing_keys)}个")
        state_dict = IF.upgrade_state_dict(state_dict, model)
        print("自适应阈值参数添加完成!")
    
    return model.load_state_dict(state_dict, strict=strict)