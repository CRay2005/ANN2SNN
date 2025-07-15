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
import pandas as pd
import os

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


class SurrogateSpike(torch.autograd.Function):
    """自定义代理梯度脉冲函数（参考gradient_cray.py的设计思路）"""
    @staticmethod
    def forward(ctx, mem_pot, threshold, surrogate_grad='arctan', scale=1.0):
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
    
myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(self, T=0, L=8, thresh=8.0, tau=1., gama=1.0, 
                surrogate_grad='sigmoid', scale=5.0, optimize_thre_flag=False):
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

        # 如果为True，则不进行自适应阈值的计算，直接使用QCFS计算
        self.optimize_thre_flag = optimize_thre_flag    

        # 初始化神经元阈值为0，在第一次forward时会初始化为合适的形状
        self.register_buffer("neuron_thre", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))
        
        # 统计信息
        self.register_buffer("update_count", torch.tensor(0))
        
        # 添加存储阈值的列表
        self.thre_history = []  # 存储每个批次的阈值
        
        # 代理梯度相关参数
        self.surrogate_grad = surrogate_grad  # 代理梯度类型
        self.scale = scale                    # 代理梯度缩放因子
        self.surrogate_spike = SurrogateSpike.apply  # 自定义代理梯度函数         

    def QCFS(self,x):
        if self.T > 0:
            thre = self.neuron_thre
            if thre.dim() == 0:
                thre = thre.unsqueeze(0)
            x = self.expand(x)
            input_channels = x.shape[2]
            
            # 初始化mem
            mem = torch.zeros_like(x[0, ...])
            
            # 根据阈值维度选择不同的处理方式
            if thre.shape[0] == 1:
                # 如果只有一个阈值，直接广播
                mem = mem + 0.5 * thre
            else:
                # 如果有多个阈值，确保维度匹配
                assert thre.shape[0] == input_channels, f"阈值通道数({thre.shape[0]})与输入通道数({input_channels})不匹配"
                mem = mem + 0.5 * thre.view(1, -1, 1, 1)
            
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                if thre.shape[0] == 1:
                    # 单阈值情况
                    spike = self.act(mem - thre, self.gama) * thre
                    # spike = self.surrogate_spike(mem, thre, self.surrogate_grad, self.scale) * thre
                else:
                    # 多阈值情况
                    spike = self.act(mem - thre.view(1, -1, 1, 1), self.gama) * thre.view(1, -1, 1, 1)
                    # spike = self.surrogate_spike(mem, thre, self.surrogate_grad, self.scale) * thre.view(1, -1, 1, 1)
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
    
    def forward(self, x):

        if not self.optimize_thre_flag:
            x=self.QCFS(x)
            return x  
             
        #下面的过程是计算每个神经元的最优阈值
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)
            # 初始化神经元级别的阈值
            if not self.initialized:
                self._initialize_neuron_thresh(x)
                self.initialized = torch.tensor(True)
            
            # # 选择使用的阈值
            # if self.enable_adaptive_thresh:
            #     # 使用自适应阈值 - 正确处理维度扩展
            #     if len(x[0].shape) == 4:  # [B, C, H, W] 卷积层
            #         current_thre = self.neuron_thre.view(1, -1, 1, 1).expand_as(x[0])
            #     elif len(x[0].shape) == 2:  # [B, C] 全连接层
            #         current_thre = self.neuron_thre.unsqueeze(0).expand_as(x[0])
            #     else:  # 其他情况使用固定阈值
            #         current_thre = thre
            # else:
            #     # 使用固定阈值
            #     current_thre = thre

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

                    # 确保存储完整的batch数据
                    thre_data = new_thre.detach().cpu().numpy()  
                    if len(thre_data.shape) == 4:  # (batch, channel, height, width)
                        for b in range(thre_data.shape[0]):
                            batch_data = thre_data[b:b+1]  # 保持4D形状
                            self.thre_history.append(batch_data)
                    else:
                        # 如果不是4D数据，扩展为4D
                        thre_data = thre_data.reshape(1, 1, *thre_data.shape) 
                        self.thre_history.append(thre_data)
                    
                    # 4.更新自适应阈值（仅在训练模式下）
                    # if self.enable_adaptive_thresh:
                    #     self._update_adaptive_thresh(new_thre)
                    
                    # 5.将扩展后的neuron_thre乘以spike_pot中的元素
                    for i in range(len(spike_pot)):
                        spike_pot[i] = spike_pot[i] * new_thre
                    
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
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
        
    def _update_adaptive_thresh(self, new_thre):
        """使用梯度下降方式更新自适应阈值"""
        if self.neuron_thre.numel() == 1:
            return
            
        # 直接使用new_thre，不进行平均
        if len(new_thre.shape) == 4:  # [B, C, H, W]
            # 如果维度不匹配，只保留通道维度
            if new_thre.shape[1] != self.neuron_thre.shape[0]:
                print(f"警告 - 通道维度不匹配: new_thre {new_thre.shape}, neuron_thre {self.neuron_thre.shape}")
                return
            current_thre = new_thre[0, :, 0, 0]  # 只取第一个批次和空间位置的值
        elif len(new_thre.shape) == 2:  # [B, C]  
            # 如果维度不匹配，只保留通道维度
            if new_thre.shape[1] != self.neuron_thre.shape[0]:
                print(f"警告 - 通道维度不匹配: new_thre {new_thre.shape}, neuron_thre {self.neuron_thre.shape}")
                return
            current_thre = new_thre[0, :]  # 只取第一个批次的值
        else:
            current_thre = new_thre
            
        # 直接更新神经元阈值，使用较小的学习率以保持稳定性
        # self.neuron_thre = self.neuron_thre + self.c * (current_thre - self.neuron_thre)
        self.neuron_thre = current_thre
        self.update_count += 1
        
    # def get_thresh_stats(self):
    #     """获取阈值统计信息"""
    #     return {
    #         'neuron_thresh': self.neuron_thre.mean().item() if self.neuron_thre.numel() > 1 else self.neuron_thre.item(),
    #         'original_thresh': self.thresh.item(),
    #         'update_count': self.update_count.item()
    #     }
    
    # def reset_adaptive_thresh(self):
    #     """重置自适应阈值到原始值"""
    #     if self.initialized:
    #         self.neuron_thre.data.fill_(self.thresh.data.item())
    #         self.thresh_diff_sum.zero_()
    #         self.thresh_diff_count.zero_()
    #         self.update_count.zero_()

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
                for buffer_name in ['update_count']:
                    buffer_key = f"{name}.{buffer_name}"
                    if buffer_key not in upgraded_state_dict:
                        if buffer_name == 'update_count':
                            upgraded_state_dict[buffer_key] = torch.tensor(0)
        
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
                for buffer_name in ['neuron_thre', 'initialized', 'update_count']:
                    key = f"{name}.{buffer_name}"
                    if key not in state_dict:
                        missing_keys.append(key)
        
        if missing_keys:
            print(f"检测到旧版本模型，自动添加缺失的参数: {len(missing_keys)}个")
            state_dict = IF.upgrade_state_dict(state_dict, self)
        
        return self.load_state_dict(state_dict, strict=strict)

    def get_channel_thresholds(self):
        """计算每个通道的阈值，只考虑非零脉冲数据"""
        if not self.thre_history:
            return None
            
        try:
            # 获取最新的阈值数据
            thre_data = self.thre_history[-1]
            
            # 确保数据是2D的 (channels, height*width)
            if len(thre_data.shape) == 4:
                thre_data = thre_data[0]
                thre_data = thre_data.reshape(thre_data.shape[0], -1)
            elif len(thre_data.shape) == 3:
                thre_data = thre_data.reshape(thre_data.shape[0], -1)
                
            # 计算每个通道的阈值（只考虑非零数据）
            channel_thresholds = []
            for channel_data in thre_data:
                # 只保留非零数据
                non_zero_data = channel_data[channel_data > 0]
                if len(non_zero_data) > 0:
                    # 使用非零数据的中位数作为通道阈值
                    channel_thresholds.append(np.median(non_zero_data))
                else:
                    # 如果通道所有数据都是0，使用默认阈值
                    channel_thresholds.append(self.thresh.item())
            
            return np.array(channel_thresholds)
            
        except Exception as e:
            print(f"计算通道阈值时出错: {str(e)}")
            return None
            
    def save_thresholds_to_csv(self, save_path):
        """保存阈值数据到CSV文件，使用精简格式，并计算每个通道的统计特征（只考虑非零数据）"""
        if not self.thre_history:
            print("没有阈值数据可保存")
            return
            
        try:
            # 获取所有batch的阈值数据
            all_thre_data = self.thre_history  # 获取所有batch的阈值数据
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存统计数据到单独的文件
            stats_path = save_path.replace('.csv', '_stats.csv')
            with open(stats_path, 'w') as f:
                # 写入统计信息表头
                f.write("通道ID,非零数据总数,均值,中位数,最大值,最小值,标准差,50分位,60分位,70分位,80分位,90分位,95分位,99分位\n")
                
                # 获取通道数量
                num_channels = all_thre_data[0].shape[1]
                
                # 对每个通道进行统计
                for channel_id in range(num_channels):
                    # 收集该通道在所有batch中的所有非零数据
                    channel_non_zero_data = []
                    total_non_zero_count = 0
                    
                    for thre_data in all_thre_data:
                        # 确保数据是4D的 (batch, channel, height, width)
                        if len(thre_data.shape) == 3:  # (channel, height, width)
                            thre_data = thre_data.reshape(1, *thre_data.shape)
                        
                        # 获取当前通道的所有数据
                        channel_data = thre_data[:, channel_id, :, :].reshape(-1)
                        non_zero_data = channel_data[channel_data > 0]
                        channel_non_zero_data.extend(non_zero_data)
                        total_non_zero_count += len(non_zero_data)
                    
                    if len(channel_non_zero_data) > 0:
                        # 计算统计特征
                        mean_val = np.mean(channel_non_zero_data)
                        median_val = np.median(channel_non_zero_data)
                        max_val = np.max(channel_non_zero_data)
                        min_val = np.min(channel_non_zero_data)
                        std_val = np.std(channel_non_zero_data)
                        
                        # 计算分位数
                        percentiles = np.percentile(channel_non_zero_data, [50, 60, 70, 80, 90, 95, 99])
                        p50, p60, p70, p80, p90, p95, p99 = percentiles
                    else:
                        # 如果所有数据都是0，所有统计值都设为0
                        mean_val = median_val = max_val = min_val = std_val = 0.0
                        p50 = p60 = p70 = p80 = p90 = p95 = p99 = 0.0
                    
                    # 写入统计信息
                    f.write(f"{channel_id},{total_non_zero_count},{mean_val:.6f},{median_val:.6f},{max_val:.6f},{min_val:.6f},{std_val:.6f},{p50:.6f},{p60:.6f},{p70:.6f},{p80:.6f},{p90:.6f},{p95:.6f},{p99:.6f}\n")
            
            # 保存整体统计信息到另一个文件
            summary_path = save_path.replace('.csv', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("阈值统计信息（只考虑非零数据）:\n")
                all_non_zero_data = []
                for thre_data in all_thre_data:
                    if len(thre_data.shape) == 3:
                        thre_data = thre_data.reshape(1, *thre_data.shape)
                    # 展平所有维度
                    thre_data = thre_data.reshape(-1)
                    non_zero_data = thre_data[thre_data > 0]
                    all_non_zero_data.extend(non_zero_data)
                
                if len(all_non_zero_data) > 0:
                    f.write(f"所有通道的平均阈值: {np.mean(all_non_zero_data):.6f}\n")
                    f.write(f"所有通道的中位数阈值: {np.median(all_non_zero_data):.6f}\n")
                    f.write(f"所有通道的最大阈值: {np.max(all_non_zero_data):.6f}\n")
                    f.write(f"所有通道的最小阈值: {np.min(all_non_zero_data):.6f}\n")
                    f.write(f"所有通道的标准差: {np.std(all_non_zero_data):.6f}\n")
                    
                    # 添加分位数统计
                    percentiles = np.percentile(all_non_zero_data, [50, 60, 70, 80, 90, 95, 99])
                    f.write(f"所有通道的50分位阈值: {percentiles[0]:.6f}\n")
                    f.write(f"所有通道的60分位阈值: {percentiles[1]:.6f}\n")
                    f.write(f"所有通道的70分位阈值: {percentiles[2]:.6f}\n")
                    f.write(f"所有通道的80分位阈值: {percentiles[3]:.6f}\n")
                    f.write(f"所有通道的90分位阈值: {percentiles[4]:.6f}\n")
                    f.write(f"所有通道的95分位阈值: {percentiles[5]:.6f}\n")
                    f.write(f"所有通道的99分位阈值: {percentiles[6]:.6f}\n")
                f.write(f"当前IF层固定阈值: {self.thresh.item():.6f}\n")
                
        except Exception as e:
            print(f"保存阈值数据时出错: {str(e)}")
            print(f"数据形状: {thre_data.shape if 'thre_data' in locals() else 'unknown'}")


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x




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
    # print("开始兼容性模型加载...")
    
    # 检查是否存在新增的buffer参数
    missing_keys = []
    for name, module in model.named_modules():
        if isinstance(module, IF):
            for buffer_name in ['neuron_thre', 'initialized', 'update_count']:
                key = f"{name}.{buffer_name}"
                if key not in state_dict:
                    missing_keys.append(key)
    
    if missing_keys:
        print(f"检测到旧版本模型，自动添加缺失的自适应阈值参数: {len(missing_keys)}个")
        state_dict = IF.upgrade_state_dict(state_dict, model)
        # print("自适应阈值参数添加完成!")
    
    return model.load_state_dict(state_dict, strict=strict)



# class GradientAnalyzer:
#     def __init__(self, model, prune_ratio=0.1):
#         self.model = model
#         self.prune_ratio = prune_ratio
#         self.gradient_records = defaultdict(list)
        
#         # 注册梯度钩子
#         for name, param in model.named_parameters():
#             if 'weight' in name:
#                 hook = self.make_hook(name)
#                 param.register_hook(hook)
    
#     def make_hook(self, name):
#         def gradient_hook(grad):
#             layer_idx = int(name.split('.')[1])  # e.g. 'layers.0.weight'
#             grad_mag = grad.abs().mean(dim=1)    # 平均输入梯度
#             self.gradient_records[layer_idx].append(grad_mag.detach().cpu())
#         return gradient_hook
    
#     def get_low_grad_neurons(self):
#         all_neuron_grads = {}
        
#         # 处理每层梯度记录
#         for layer_idx, grads in self.gradient_records.items():
#             time_avg_grad = torch.stack(grads).mean(dim=0)  # 时间维度平均
            
#             # 神经元梯度统计 (公式1)
#             neuron_grads = time_avg_grad.mean(dim=0)  # [神经元] <- 批次平均
            
#             # 保存神经元索引和梯度值
#             for neuron_id, grad_val in enumerate(neuron_grads):
#                 key = (layer_idx, neuron_id)
#                 all_neuron_grads[key] = grad_val.item()
        
#         # 筛选梯度最小的神经元 (公式2)
#         sorted_neurons = sorted(all_neuron_grads.items(), key=lambda x: x[1])
#         num_select = int(len(sorted_neurons) * self.prune_ratio)
#         return sorted_neurons[:num_select]  # [(层ID, 神经元ID), ...]



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

