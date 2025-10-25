'''
        #第一步粗略调整threshold,适当上调thre使得脉冲能在T时间发放完成
        ub = self.thre
        #把T时间的x平均之后与thre比较,thre = thre + k * ((x_average) - thre) * (x_average > thre)
        #x_average = x.mean(0, keepdim=True)
        #v(T)-v(0)/T = thre - x_average
        #thre = thre + k * (v(T)-v(0)/T - thre) * (v(T)-v(0)/T > thre)
        #thre = thre + k * (v(T)-v(0)/T - thre) * (v(T)-v(0)/T > thre)
        #第二步用类似关于误差的rnn精细训练thre
        #v(T)-v(0)/T要转化为关于thre的式子，通过第一步可以理想估计发送脉冲数为(x/thre取整)
        #故 v(T)-v(0)/T = (x - 取整x/thre * thre)/T,thre微调过程可假设x/thre取整是常数
        
        #第三步剪枝,去除误差仍很大或响应很小的神经元
'''

from cv2 import mean
from sympy import print_rcode
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import math

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

class SurrogateSpike(torch.autograd.Function):
    """自定义代理梯度脉冲函数（参考gradient_cray.py的设计思路）"""
    @staticmethod
    def forward(ctx, mem_pot, threshold, surrogate_grad='arctan', scale=5.0):
        # 计算脉冲
        spike = (mem_pot >= threshold).float()
        
        # 保存用于反向传播的信息
        delta = mem_pot - threshold
        ctx.save_for_backward(delta)
        ctx.surrogate_grad = surrogate_grad
        ctx.scale = scale
        
        return spike
    
    @staticmethod
    def backward(ctx, grad_output):
        """自定义反向传播：应用代理梯度"""
        delta = ctx.saved_tensors[0]
        surrogate_grad = ctx.surrogate_grad
        scale = ctx.scale
        
        # 计算代理梯度
        if surrogate_grad == 'sigmoid':
            sg = torch.sigmoid(scale * delta)
            surrogate_gradient = sg * (1 - sg) * scale
        elif surrogate_grad == 'arctan':
            surrogate_gradient = scale / (1 + (scale * np.pi * delta) ** 2)
        elif surrogate_grad == 'triangle':
            surrogate_gradient = torch.clamp(1 - scale * torch.abs(delta), min=0)
        else:  # 默认使用sigmoid
            sg = torch.sigmoid(scale * delta)
            surrogate_gradient = sg * (1 - sg) * scale
        
        # 应用代理梯度到上游梯度
        modified_grad = grad_output * surrogate_gradient
        
        # 返回梯度：对mem_pot的梯度, 对threshold的负梯度, surrogate_grad参数, scale参数
        # delta = mem_pot - threshold，所以∂L/∂threshold = -∂L/∂delta
        threshold_grad = -modified_grad
        
        return modified_grad, threshold_grad, None, None

class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, layer_name="IF", 
                 surrogate_grad='sigmoid', scale=5, num_neurons=0):
        super(IF, self).__init__()
        self.act = ZIF.apply
        # 强制使用标量阈值，忽略num_neurons参数
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0
        self.layer_name = layer_name
        self.surrogate_grad = surrogate_grad  # 代理梯度类型
        self.scale = scale                    # 代理梯度缩放因子
        self.surrogate_spike = SurrogateSpike.apply  # 自定义代理梯度函数 
        self.delta = 0                        
        # 初始化 critical_count 为 None，在第一次forward时会根据输入形状调整
        # 使用 None 而不是空张量，避免状态保存/恢复时的形状冲突
        self._critical_count = None
    

        self.spike_count = 0
        self.spike_count_tensor = None  # 延迟初始化，在forward中根据输入形状设置
        self.mem = 0.5 * self.thresh
        self.predict_spike = torch.tensor(-1.0)
        self.record_spike = torch.tensor(0.0)
        # 是否记录时间维上的膜电位历史（仅在需要时开启，避免额外内存开销）
        self.record_mem_history = False
        self.mem_history = []
    
    @property
    def critical_count(self):
        """动态获取 critical_count，如果未初始化则返回 None"""
        return self._critical_count
    
    @critical_count.setter
    def critical_count(self, value):
        """设置 critical_count"""
        self._critical_count = value
    
    def reset_state(self):
        """重置IF层状态，避免外部直接修改破坏梯度连接"""
        with torch.no_grad():
            if self.thresh.numel() == 1:
                self.mem = 0.5 * self.thresh
            else:
                # 向量阈值情况，使用平均值初始化
                self.mem = 0.5 * self.thresh.mean()
            
            if isinstance(self.spike_count_tensor, torch.Tensor):
                self.spike_count_tensor.zero_()
            
            self.predict_spike = torch.tensor(-1.0)
            self.record_spike = torch.tensor(0.0)
    
    def set_T(self, T):
        """设置时间步，并同步到展开/合并模块"""
        self.T = T
        # 同步时间步到时间维处理模块
        if hasattr(self, 'expand') and isinstance(self.expand, ExpandTemporalDim):
            self.expand.T = T
        if hasattr(self, 'merge') and isinstance(self.merge, MergeTemporalDim):
            self.merge.T = T
        return

    def forward(self, x):
        if self.T > 0:
            thre = self.thresh  # 保持梯度连接，不使用.data
            x = self.expand(x)
            
            # 初始化膜电位 - 修复版本，让阈值调整真正生效
            if thre.numel() == 1:
                mem = 0.5 * thre  # 标量阈值，广播
            else:
                # 向量阈值，需要reshape到正确的形状
                input_shape = x[0, ...].shape[1:]  # 去掉batch维度
                if thre.numel() == input_shape.numel():
                    # 阈值向量大小与输入匹配，直接reshape
                    thre_reshaped = thre.view(input_shape).unsqueeze(0).expand_as(x[0, ...])
                    mem = 0.5 * thre_reshaped
                else:
                    # 阈值向量大小不匹配，使用第一个值
                    mem = 0.5 * thre[0]
            
            # 按当前批次的形状初始化计数张量，确保与 spike_binary 对齐
            # 只在第一次或形状变化时重新初始化
            if self.spike_count_tensor is None or self.spike_count_tensor.shape != x[0, ...].shape:
                self.spike_count_tensor = torch.zeros_like(x[0, ...])
            spike_pot = []

            # QCFC处理 - 使用模拟梯度============================================
                        
            for t in range(self.T):
                mem = mem + x[t, ...] 
                if t == 0:
                    if self.critical_count is None:
                        # 确保获取正确的神经元数量，处理多维情况
                        if mem.dim() > 2:
                            # 如果是卷积层输出，需要flatten后获取神经元数量
                            num_neurons = mem.numel() // mem.shape[0]  # 总元素数除以batch_size
                        else:
                            num_neurons = mem.shape[-1]
                        self.critical_count = torch.zeros(num_neurons, device=mem.device)
                        print(f"IF层 {self.layer_name}: 初始化 critical_count 大小为 {num_neurons}")
                    
                    # 初始化时间累计计数器（按神经元维度）
                    if mem.dim() > 2:
                        # 对于多维情况，flatten后计算
                        temporal_critical_count = torch.zeros(mem.numel() // mem.shape[0], device=mem.device)
                    else:
                        temporal_critical_count = torch.zeros(mem.shape[-1], device=mem.device)
                    
                if self.record_mem_history:
                    # 记录当前时间步的膜电位（不打断计算图）
                    self.mem_history.append(mem.detach().cpu())
                
                # 检查临界状态 - 修复版本
                if thre.numel() == 1:
                    is_critical = torch.abs(mem - thre) < (0.1 * thre)
                else:
                    # 向量阈值，需要reshape到正确的形状
                    mem_shape = mem.shape[1:]
                    if thre.numel() == mem_shape.numel():
                        thre_reshaped = thre.view(mem_shape).unsqueeze(0).expand_as(mem)
                        is_critical = torch.abs(mem - thre_reshaped) < (0.1 * thre_reshaped)
                    else:
                        is_critical = torch.abs(mem - thre[0]) < (0.1 * thre[0])
                
                # 处理多维情况：flatten后计算平均值
                if mem.dim() > 2:
                    # 将多维张量flatten为 [batch_size, num_neurons]
                    batch_size = mem.shape[0]
                    mem_flat = mem.view(batch_size, -1)
                    is_critical_flat = is_critical.view(batch_size, -1)
                    # 沿batch维度取平均
                    temporal_critical_count += is_critical_flat.float().mean(dim=0)
                else:
                    # 沿batch维度取平均，然后累加到时间计数器上
                    # 这代表了"平均一个样本"在当前时间步t有多少神经元是临界的
                    temporal_critical_count += is_critical.float().mean(dim=0)

                # 计算脉冲 - 修复版本
                if thre.numel() == 1:
                    spike_binary = self.surrogate_spike(mem, thre, self.surrogate_grad, self.scale)
                    spike = spike_binary * thre
                else:
                    # 向量阈值，需要reshape到正确的形状
                    mem_shape = mem.shape[1:]
                    if thre.numel() == mem_shape.numel():
                        thre_reshaped = thre.view(mem_shape).unsqueeze(0).expand_as(mem)
                        spike_binary = self.surrogate_spike(mem, thre_reshaped, self.surrogate_grad, self.scale)
                        spike = spike_binary * thre_reshaped
                    else:
                        spike_binary = self.surrogate_spike(mem, thre[0], self.surrogate_grad, self.scale)
                        spike = spike_binary * thre[0]
                
                self.spike_count_tensor = self.spike_count_tensor + spike_binary.float()
                mem = mem - spike
                spike_pot.append(spike)
                

                # 预测脉冲
                self.predict_spike = torch.where(spike_binary == 1, 0.0, self.predict_spike)
                self.predict_spike = torch.where(self.record_spike == 1, -1.0, self.predict_spike)
                
                # 预测脉冲的阈值比较 - 修复版本
                if thre.numel() == 1:
                    thre_for_pred = thre
                else:
                    mem_shape = mem.shape[1:]
                    if thre.numel() == mem_shape.numel():
                        thre_for_pred = thre.view(mem_shape).unsqueeze(0).expand_as(mem)
                    else:
                        thre_for_pred = thre[0]
                
                self.predict_spike = torch.where(torch.logical_and(mem < thre_for_pred * 0.1, self.predict_spike == -1.0), -1.0, torch.maximum(self.predict_spike, torch.tensor(0.0)))
                self.predict_spike = torch.where(mem > thre_for_pred * 0.9, 1.0, self.predict_spike)
                
                self.record_spike = spike_binary.detach()
            
             
            # =======================================================
            
            # 将时间累计的临界计数更新到 critical_count
            if self.critical_count is not None:
                self.critical_count += temporal_critical_count

            self.mem = mem
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)

        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh
        return x

def add_dimention(x, T):
    x = x.unsqueeze(0)  # 添加时间维度
    x = x.repeat(T, 1, 1, 1, 1)  # 重复T次
    return x


