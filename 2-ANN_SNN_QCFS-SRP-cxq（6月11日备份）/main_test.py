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
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
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
    model.set_L(4)

    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)
    
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