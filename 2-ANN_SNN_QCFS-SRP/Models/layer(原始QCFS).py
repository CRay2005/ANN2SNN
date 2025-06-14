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
            
            # QCFC处理============================================
                        
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = self.act(mem - thre, self.gama) * thre
                mem = mem - spike
                spike_pot.append(spike)
            # =======================================================



            # # 模电压补偿处理============================================
            # mem = 0.5 * thre  # 初始化膜电位            
            # # mem = torch.zeros_like(x[0, ...])           

            # for t in range(self.T):
            #     mem = mem + x[t, ...]
            #     # spike = self.act(mem - thre, self.gama) * thre
            #     spike = self.act(mem - thre, self.gama) 
            #     # 处理dead neuron
            #     # spike = spike * deadneuron_flag  
            #     mem = mem - spike* thre
            #     spike_pot.append(spike)


            #     if t == self.T-1  :

            #         compen_mem = (mem - thre/2)
                    
            #         # 1.计算mem对应的spike的数量
            #         spike_count = torch.stack(spike_pot, dim=0).sum(dim=0)  # 计算每个位置在所有时间步的spike总数
                    
            #         # 2.如果（compen_mem + spike_count*thre）>0,
            #         # 则new_thre=（compen_mem + spike_count*thre）/（spike_count），
            #         # 否则，new_thre=0
            #         # 增加判断：如果(compen_mem + spike_count * thre) > self.T*thre，则取值为self.T*thre
            #         compen_value = compen_mem + spike_count * thre
            #         compen_value = torch.where(
            #             compen_value > self.T * thre,
            #             self.T * thre,
            #             compen_value
            #         )

            #         # spike_count_safe = torch.where(spike_count > 0, spike_count, torch.ones_like(spike_count))  # 避免除零
            #         new_thre = torch.where(
            #             (compen_value > 0) & (spike_count > 0),
            #             compen_value / spike_count,
            #             torch.zeros_like(compen_mem)
            #         )
                    
            #         # 3.将new_thre乘以spike_pot中的元素
            #         for i in range(len(spike_pot)):
            #             spike_pot[i] = spike_pot[i] * new_thre
                    
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
    
    
    #加入模拟梯度
    @staticmethod
    def backward(ctx, grad_output):
        """自定义反向传播：应用模拟梯度"""
        # 获取前向传播保存的值
        delta = ctx.saved_tensors[0]
        scale = ctx.scale
        surrogate_grad = ctx.surrogate_grad
        
        # 计算模拟梯度
        if surrogate_grad == 'sigmoid':
            sg = torch.sigmoid(scale * delta)
            surrogate_grad = sg * (1 - sg) * scale
        elif surrogate_grad == 'arctan':
            surrogate_grad = scale / (1 + (scale * np.pi * delta) ** 2)
        elif surrogate_grad == 'triangle':
            surrogate_grad = torch.clamp(1 - scale * torch.abs(delta), min=0)
        else:  # 默认使用sigmoid
            sg = torch.sigmoid(scale * delta)
            surrogate_grad = sg * (1 - sg) * scale
        
        # 应用模拟梯度到上游梯度
        modified_grad = grad_output * surrogate_grad
        
        return modified_grad, None, None, None, None

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


class SNNGradientSimulator:
    def __init__(self, model, surrogate_fn=None):
        """
        SNN模拟梯度反向传播工具
        
        参数:
        model -- SNN模型
        surrogate_fn -- 自定义模拟梯度函数，默认为sigmoid梯度近似
        """
        self.model = model
        self.gradient_records = defaultdict(list)
        self.activations = {}  # 存储前向激活值
        
        # 设置默认的模拟梯度函数
        self.surrogate_grad = self.default_surrogate_grad
        
        # 注册前向和反向钩子
        self.register_hooks()
    
    def default_surrogate_grad(self, x):
        """梯度函数"""
        sg = torch.sigmoid(5 * x)  # 使用5倍的斜度以获得更尖锐的梯度
        return sg * (1 - sg)
    
    def register_hooks(self):
        """为全连接层注册钩子"""
        # 清除所有现有钩子
        self.handles = []
        
        # 注册前向钩子以存储激活值
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):  # 仅处理Linear层
                # 前向钩子存储激活值
                forward_hook = self.make_forward_hook(name)
                handle = module.register_forward_hook(forward_hook)
                self.handles.append(handle)
                
                # 反向钩子应用模拟梯度
                backward_hook = self.make_backward_hook(name)
                handle = module.register_full_backward_hook(backward_hook)
                self.handles.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def make_forward_hook(self, name):
        """创建前向钩子"""
        def forward_hook(module, input, output):
            # 存储当前模块的激活值（用于模拟梯度计算）
            self.activations[name] = output.detach().clone()
        return forward_hook
    
    def make_backward_hook(self, name):
        """创建反向钩子（应用模拟梯度）"""
        def backward_hook(module, grad_input, grad_output):
            # 跳过输入层
            if grad_output[0] is None:
                return
            
            # 获取模块激活值（如果没有记录，则返回原梯度）
            if name not in self.activations:
                return
            
            # 计算模拟梯度
            activations = self.activations[name]
            surrogate = self.surrogate_grad(activations)
            
            # 应用模拟梯度到上游梯度
            modified_grad_output = []
            for grad_out in grad_output:
                if grad_out is not None:
                    # 确保梯度和模拟梯度形状兼容
                    if grad_out.shape == surrogate.shape:
                        modified_grad_out = grad_out * surrogate
                    else:
                        # 处理形状不匹配情况（例如卷积层）
                        # 这里只是简单缩放梯度，可能需要根据具体情况调整
                        modified_grad_out = grad_out * surrogate.mean()
                    modified_grad_output.append(modified_grad_out)
                else:
                    modified_grad_output.append(None)
            
            # 返回修改后的梯度
            return tuple(modified_grad_output)
        
        return backward_hook
    
    def analyze_fc_gradients(self, prune_ratio=0.1):
        """
        分析全连接层梯度并返回低梯度神经元
        必须先运行反向传播才能调用此方法
        
        参数:
        prune_ratio -- 剪枝比例，默认0.1
        
        返回:
        低梯度神经元列表 (层名, 神经元索引)
        """
        # 收集全连接层的梯度信息
        fc_gradients = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                # 获取权重梯度
                weight_grad = module.weight.grad.detach()
                
                # 计算每个输出神经元的平均梯度
                # 对于全连接层，对输入维度取平均
                neuron_grads = weight_grad.abs().mean(dim=1)
                
                fc_gradients[name] = {
                    'gradients': neuron_grads,
                    'module': module
                }
        
        # 如果没有找到全连接层梯度，返回空列表
        if not fc_gradients:
            return []
        
        # 聚合所有神经元梯度
        all_neurons = []
        for layer_name, data in fc_gradients.items():
            grads = data['gradients']
            module = data['module']
            
            for neuron_idx in range(module.out_features):
                grad_value = grads[neuron_idx].item()
                all_neurons.append((layer_name, neuron_idx, grad_value))
        
        # 按梯度值排序（从低到高）
        all_neurons.sort(key=lambda x: x[2])
        
        # 选择梯度最小的神经元
        num_prune = int(len(all_neurons) * prune_ratio)
        return all_neurons[:num_prune]
    
    def prune_neurons(self, neurons_to_prune):
        """剪枝低梯度神经元"""
        for layer_name, neuron_idx, _ in neurons_to_prune:
            # 找到对应模块
            module = None
            for name, mod in self.model.named_modules():
                if name == layer_name and isinstance(mod, nn.Linear):
                    module = mod
                    break
            
            if module is None:
                continue
                
            # 执行剪枝（将对应神经元的权重置零）
            with torch.no_grad():
                # 剪枝输出权重
                module.weight.data[neuron_idx] = 0
                
                # 如果有偏置项，剪枝偏置
                if module.bias is not None:
                    module.bias.data[neuron_idx] = 0
                
                # 剪枝下游连接（针对下一层的输入权重）
                self.prune_downstream_connections(module, neuron_idx)
    
    def prune_downstream_connections(self, pruned_module, pruned_neuron_idx):
        """剪枝下游层的输入连接"""
        # 查找所有依赖该模块输出的下游模块
        downstream_modules = self.find_downstream_modules(pruned_module)
        
        for module in downstream_modules:
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # 剪枝对应输入权重（列对应输入神经元）
                    module.weight.data[:, pruned_neuron_idx] = 0
    
    def find_downstream_modules(self, target_module):
        """查找所有直接使用目标模块输出的下游模块"""
        downstream = []
        
        # 简单实现：假设直接下游层是model._modules中位置后面的层
        # 在实际应用中，可能需要更复杂的依赖关系分析
        found = False
        for module in self.model.modules():
            if module is target_module:
                found = True
            elif found:
                downstream.append(module)
        
        return downstream
    
    

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class SNNGradientAnalyzer:
    """SNN梯度分析工具：模拟梯度反向传播与梯度统计"""
    def __init__(self, model, surrogate_grad_type='sigmoid', grad_scale=5.0):
        """
        初始化SNN梯度分析器
        
        参数:
        model - 已训练好的SNN模型
        surrogate_grad_type - 模拟梯度类型: 'sigmoid', 'arctan', 'triangle'
        grad_scale - 模拟梯度缩放因子
        """
        self.model = model
        self.surrogate_grad_type = surrogate_grad_type
        self.grad_scale = grad_scale
        self.gradient_hooks = {}
        self.gradient_records = defaultdict(list)
        self.forward_cache = {}
        
        # 为所有激活层(IF神经元)注册前向钩子
        self.register_activation_hooks()
    
    def apply_surrogate_grad(self, x):
        """应用模拟梯度函数"""
        if self.surrogate_grad_type == 'sigmoid':
            sg = torch.sigmoid(self.grad_scale * x)
            return sg * (1 - sg)
        elif self.surrogate_grad_type == 'arctan':
            return 1 / (1 + (self.grad_scale * np.pi * x)**2)
        elif self.surrogate_grad_type == 'triangle':
            return torch.clamp(1 - self.grad_scale * torch.abs(x), min=0)
        else:
            # 默认使用sigmoid
            sg = torch.sigmoid(self.grad_scale * x)
            return sg * (1 - sg)
    
    def register_activation_hooks(self):
        """为所有激活层注册前向钩子"""
        for name, module in self.model.named_modules():
            # 识别SNN中的激活层（IF神经元）
            if 'IF' in module.__class__.__name__ or 'LIF' in module.__class__.__name__:
                # 注册前向钩子记录激活值
                hook = self._forward_hook(name)
                handle = module.register_forward_hook(hook)
                self.gradient_hooks[name] = handle
    
    def _forward_hook(self, name):
        """创建前向钩子函数"""
        def hook(module, input, output):
            # 记录激活值(模拟梯度计算依据)
            self.forward_cache[name] = output.detach().clone()
        return hook
    
    def backward_with_surrogate(self, output, target, criterion):
        """使用模拟梯度进行反向传播"""
        # 清空前向缓存
        self.forward_cache = {}
        
        # 前向传播（激活钩子将被触发）
        pred = output if output.dim() <= 2 else output.mean(2)  # 处理时序输出
        loss = criterion(pred, target)
        
        # 获取梯度函数
        grad_fn = loss.grad_fn
        
        # 修改梯度函数链
        def custom_grad(grad_output):
            # 原始梯度
            orig_grad = grad_output.clone()
            
            # 在梯度回传过程中应用模拟梯度
            for name, module in self.model.named_modules():
                # 只在激活层应用模拟梯度
                if name in self.forward_cache:
                    # 获取原始梯度
                    if orig_grad.grad_fn.next_functions:
                        # 获取对应层的梯度
                        grad_input = orig_grad.grad_fn.next_functions[0][0]
                        
                        # 应用模拟梯度
                        cached_value = self.forward_cache[name]
                        surrogate_grad = self.apply_surrogate_grad(cached_value)
                        
                        # 修改梯度
                        modified_grad = grad_input * surrogate_grad
                        
                        # 替换为修改后的梯度
                        grad_input.copy_(modified_grad)
            
            return orig_grad
        
        # 将自定义梯度函数附加到原始梯度上
        grad_fn.register_hook(custom_grad)
        
        # 执行反向传播（将触发自定义梯度计算）
        loss.backward()
        
        return loss
    
    def register_fc_gradient_hooks(self):
        """为所有全连接层注册梯度钩子"""
        # 清除现有钩子
        for handle in self.gradient_hooks.values():
            handle.remove()
        self.gradient_hooks = {}
        self.gradient_records.clear()
        
        # 为所有全连接层注册梯度钩子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 注册梯度钩子
                hook = self._gradient_hook(name)
                handle = module.weight.register_hook(hook)
                self.gradient_hooks[name] = handle
    
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
            self.gradient_records[name].append(neuron_grads.detach().cpu().numpy())
        return hook
    
    def analyze_fc_gradients(self, dataloader, criterion):
        """
        分析全连接层梯度
        
        参数:
        dataloader - 数据加载器
        criterion - 损失函数
        
        返回:
        layer_grads - 每层的平均梯度统计
        """
        # 确保梯度钩子已注册
        self.register_fc_gradient_hooks()
        
        # 清除梯度记录
        self.gradient_records = defaultdict(list)
        
        # 配置模型为评估模式但保留梯度
        self.model.eval()
        
        # 处理整个数据集
        with torch.set_grad_enabled(True):
            for inputs, targets in dataloader:
                # 清空梯度
                self.model.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 使用模拟梯度进行反向传播
                loss = self.backward_with_surrogate(outputs, targets, criterion)
                
                # 打印进度
                if hasattr(dataloader, '__len__'):
                    progress = len(self.gradient_records) / len(dataloader) * 100
                    print(f"Processing: {progress:.1f}%", end='\r')
        
        # 计算平均梯度
        layer_grads = {}
        for name, grad_list in self.gradient_records.items():
            if grad_list:
                # 合并所有批次的梯度
                all_grads = np.concatenate(grad_list, axis=0)
                
                # 计算每个神经元的平均梯度
                neuron_avg_grads = np.mean(all_grads, axis=0)
                
                # 存储结果
                layer_grads[name] = {
                    'mean_grad': np.mean(neuron_avg_grads),
                    'min_grad': np.min(neuron_avg_grads),
                    'max_grad': np.max(neuron_avg_grads),
                    'neuron_grads': neuron_avg_grads,
                    'num_neurons': len(neuron_avg_grads)
                }
        
        return layer_grads
    
    def visualize_gradients(self, layer_grads):
        """可视化梯度分布"""
        plt.figure(figsize=(15, 10))
        
        # 为每层创建子图
        for i, (layer_name, grad_info) in enumerate(layer_grads.items()):
            plt.subplot(3, 3, i+1)
            
            # 获取神经元梯度
            neuron_grads = grad_info['neuron_grads']
            
            # 绘制直方图
            plt.hist(neuron_grads, bins=30, alpha=0.7)
            
            # 添加统计信息
            plt.title(f"{layer_name}\nMean: {grad_info['mean_grad']:.4f}")
            plt.xlabel("Gradient Magnitude")
            plt.ylabel("Number of Neurons")
            
            # 最多显示9层
            if i+1 >= 9:
                break
        
        plt.tight_layout()
        plt.savefig("snn_gradient_distribution.png")
        plt.close()
        
        # 打印整体统计
        print("\n=== SNN Gradient Analysis Summary ===")
        for layer_name, grad_info in layer_grads.items():
            print(f"Layer: {layer_name}")
            print(f"  Neurons: {grad_info['num_neurons']}")
            print(f"  Mean Gradient: {grad_info['mean_grad']:.6f}")
            print(f"  Min Gradient: {grad_info['min_grad']:.6f}")
            print(f"  Max Gradient: {grad_info['max_grad']:.6f}")
            print("-" * 40)
        
        return layer_grads


