

from cv2 import mean
from sympy import print_rcode
import torch
import torch.nn as nn

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
        print(f"ZIF backward\n")
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
            mem = 0.5 * thre  
            # 初始化标志位，用于判断是否产生负脉冲。
            # 注意要把T的向量位减掉，否则和下面spike的向量形式不一样
            flag = torch.zeros_like(x[0,...])
            
            for t in range(self.T):
                mem = mem + x[t, ...]
                posspike = self.act(mem - thre, self.gama) * thre
                # 添加负脉冲计算，
                # 直接取了thre的负值，是否应该和正脉冲的thre值一样，在act中进行backword？
                negspike=mem.le(thre*(-1)).float() * thre*(-1)    
                # 根据标志位确定是否产生负脉冲
                compare = torch.where(flag > 0, torch.ones_like(flag),torch.zeros_like(flag))
                negspike = negspike * compare
                # 综合正负脉冲得到最终脉冲向量
                spike = posspike + negspike
                # 处理dead neuron
                spike = spike * deadneuron_flag  
                mem = mem - spike   
                # 更新标志位
                flag= flag + spike
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
