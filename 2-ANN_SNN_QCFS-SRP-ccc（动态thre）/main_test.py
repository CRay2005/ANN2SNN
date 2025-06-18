import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer import *
from Models.layer import load_model_compatible
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

# test configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=4, type=int, help='snn simulation time')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size)   # 训练时，batch_size=200
    # train_loader, test_loader = datapool(args.dataset, 1)   # 临时测试时，batch_size=1
    # preparing model
    model = modelpool(args.model, args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    
    # if old version state_dict
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
    
    # model.load_state_dict(state_dict)
    
    # 使用兼容性加载函数
    try:
        load_model_compatible(model, state_dict)
    except Exception as e:
        print(f"兼容性加载失败，尝试常规加载: {e}")
        model.load_state_dict(state_dict, strict=False)  # 使用非严格模式作为备选

    model.to(device)

    model.set_T(args.time)
    model.set_L(8)

    # 先进行dummy forward，确保neuron_thre被正确初始化
    # print("进行dummy forward初始化neuron_thre...")
    # dummy_input = torch.zeros(1, 3, 32, 32, device=device)
    # with torch.no_grad():
    #     model(dummy_input)
    
    # 读取阈值文件并设置给IF层
    try:
        # 遍历模型中的所有模块
        if_count = 0
        for name, module in model.named_modules():
            if isinstance(module, IF):
                # 构建阈值文件路径
                thre_file = f'/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/log/cifar100_T4_b1_99/IF_{if_count}_thresholds_stats.csv'
                
                if os.path.exists(thre_file):
                    # print(f"正在读取阈值文件: {thre_file}")
                    # 读取CSV文件
                    thre_df = pd.read_csv(thre_file)
                    num_channels_from_file = len(thre_df)  # 从文件中获取通道数
                    # print(f"从阈值文件中读取到{num_channels_from_file}个通道的阈值")
                    
                    # 设置阈值
                    if if_count < 12:  # 前12层是卷积层
                        # 直接使用统计的阈值
                        module.neuron_thre = torch.tensor(
                            thre_df['均值'].values,
                            dtype=module.thresh.dtype,
                            device=module.thresh.device
                        )
                        # print(f"已成功设置第{if_count}层卷积IF的{len(thre_df)}个通道的阈值")
                    else:  # 最后3层是全连接层
                        # 对于全连接层，使用均值作为统一阈值
                        mean_threshold = thre_df['均值'].mean()
                        module.neuron_thre = torch.tensor(
                            [mean_threshold],
                            dtype=module.thresh.dtype,
                            device=module.thresh.device
                        )
                        # print(f"已成功设置第{if_count}层全连接IF的阈值为: {mean_threshold:.6f}")
                else:
                    # print(f"警告：阈值文件不存在: {thre_file}")
                    # 如果文件不存在，使用原有thresh
                    module.neuron_thre = module.thresh.clone()
                    # print(f"已设置层 {name} 的阈值为原有thresh值: {module.thresh.item():.6f}")
                
                if_count += 1
    except Exception as e:
        print(f"设置阈值时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 测试模型
    print("开始测试模型...")

    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)
    # 设置第一层IF的标记
    # first_if_found = False
    # if_count = 0
    # for m in model.modules():
    #     if isinstance(m, IF):
    #         if_count += 1
    #         if not first_if_found:
    #             m.is_first_layer = True
    #             first_if_found = True
    #             print(f"找到第一层IF，层名: {m.layer_name}")
    #         else:
    #             m.is_first_layer = False
    # print(f"总共找到 {if_count} 个IF层")

    # 统计到的15个IF层的AvgMem平均值
    # avg_mem_values = [-0.0992, -0.4381, -0.3367, -0.5063, -0.3892, -0.4605, -0.4597, 
    #                   -0.3206, -0.2139, -0.0035, 0.0306, 0.0407, 0.2094, 0.1710, 0.2007]
    # avg_mem_values = [-0.0992, -0.4381, -0.3367, -0.5063, -0.3892, -0.4605, -0.4597, 
    #                   -0.3206, -0.2139, -0.0035, 0.0306, 0.0407, 0.2094, 0.1710, 0.2007]
    # avg_mem_values = [x * 0.15 for x in avg_mem_values]
    # # 收集IF层的阈值并更新
    # if_modules = []
    # if_thresholds = []
    # for m in model.modules():
    #     if isinstance(m, IF):
    #         if_modules.append(m)
    #         if_thresholds.append(m.thresh.item())
    # #         print(f"原始IF阈值: {m.thresh.item():.4f}")
    
    # # print(f"\n找到{len(if_thresholds)}个IF层")
    # # print("阈值减去AvgMem平均值的结果并更新:")
    # for i, (m, thresh, avg_mem) in enumerate(zip(if_modules, if_thresholds, avg_mem_values)):
    #     result = thresh + avg_mem
    #     # 更新阈值参数，确保在正确的设备上
    #     m.thresh.data = torch.tensor([result], device=m.thresh.device)
    #     # print(f"IF{i+1}: {thresh:.4f} - ({avg_mem:.4f}) = {result:.4f} (已更新)")
    
    # print("\n更新后的IF阈值:")
    # for i, m in enumerate(if_modules):
    #     print(f"IF{i+1}: {m.thresh.item():.4f}")

    acc = val(model, test_loader, device, args.time)
    print(acc)

if __name__ == "__main__":
    main()