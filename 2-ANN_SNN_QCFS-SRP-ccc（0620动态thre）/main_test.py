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

# 设置环境变量抑制cuDNN警告
os.environ['CUDNN_V8_API_DISABLED'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
# 抑制PyTorch相关警告
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')    #imagenet, cifar10, cifar100
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')    #resnet18，resnet20，resnet34，vgg16 
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

    # # 生成每层神经元的最优阈值
    # print("生成每层神经元的最优阈值...")
    # acc = val(model, test_loader, device, args.time, optimize_thre_flag=True)

    # 读取阈值文件并设置给IF层
    if_count = 0
    for name, module in model.named_modules():
        if isinstance(module, IF):
            # 构建阈值文件路径
            thre_file = f'/root/autodl-tmp/0-ANN2SNN-Allinone/2-ANN_SNN_QCFS-SRP-ccc（动态thre）/log/IF_{if_count}_thresholds_stats.csv'
            
            if os.path.exists(thre_file):
                # 读取CSV文件
                thre_df = pd.read_csv(thre_file)
                
                # 设置阈值 - 根据通道数自动判断
                threshold_values = thre_df['50分位'].values if len(thre_df) > 1 else [thre_df['50分位'].mean()]
                
                # 统一创建tensor
                module.neuron_thre = torch.tensor(
                    threshold_values,
                    dtype=module.thresh.dtype,
                    device=module.thresh.device
                )
                
                # 打印信息
                layer_type = "卷积" if len(threshold_values) > 1 else "全连接"
                print(f"已成功设置第{if_count}层{layer_type}IF的{len(threshold_values)}个通道的阈值")
            else:
                # 如果文件不存在，使用原有thresh
                module.neuron_thre = module.thresh.clone()
                # print(f"已设置层 {name} 的阈值为原有thresh值: {module.thresh.item():.6f}")
            
            if_count += 1
    
    # 测试模型
    print("开始测试模型...")
    acc = val(model, test_loader, device, args.time,optimize_thre_flag=False)
    print(acc)

if __name__ == "__main__":
    main()