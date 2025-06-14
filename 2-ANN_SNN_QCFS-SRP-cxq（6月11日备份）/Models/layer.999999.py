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
    def forward(ctx, mem_pot, threshold, surrogate_grad='sigmoid', scale=5.0):
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
                 surrogate_grad='triangle', scale=5.0):
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
        self.surrogate_grad = surrogate_grad  # 代理梯度类型
        self.scale = scale                    # 代理梯度缩放因子
        self.surrogate_spike = SurrogateSpike.apply  # 自定义代理梯度函数 
    


    def forward(self, x):
        if self.T > 0:
            thre = self.thresh  # 保持梯度连接，不使用.data
            x = self.expand(x)
            mem = 0.5 * thre  # 初始化膜电位
            spike_pot = []
            
            # QCFC处理 - 使用模拟梯度技巧============================================
                        
            for t in range(self.T):
                mem = mem + x[t, ...]
                
                # === 使用自定义代理梯度函数（参考gradient_cray.py思路） ===
                spike_binary = self.surrogate_spike(mem, thre, self.surrogate_grad, self.scale)
                #spike_binary = self.act(mem-thre,self.gama)

                spike = spike_binary * thre
                
                # 膜电位重置（保持梯度连接）
                mem = mem - spike_binary * thre
                spike_pot.append(spike)
            # =======================================================

                   
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


