"""
主训练脚本 - 支持标准训练和阈值自适应训练

使用示例:
1. 标准训练:
   python main_train.py --dataset cifar10 --model vgg16 --epochs 100

2. 阈值自适应训练:
   python main_train.py --dataset cifar10 --model vgg16 --epochs 100 --use_thre_training --adjust_interval 50 --probe_num_batches 3 --adjust_scale_factor 0.005

3. 阈值自适应训练（返回梯度历史）:
   python main_train.py --dataset cifar10 --model vgg16 --epochs 100 --use_thre_training --return_thre_grads
"""

import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
import sys
import os
# 添加Models目录到路径，确保能导入layer_copy
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))
# 导入layer_copy中的IF类，替换原来的IF
from layer_copy import IF
# 将IF类注入到Models模块中
import Models
# 直接替换Models模块中的IF类
Models.IF = IF
from Preprocess import datapool
from utils import train, val, seed_all, get_logger, train_with_thre

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers', default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=300, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix', default='cray-grad', type=str,help='suffix')
parser.add_argument('-T', '--time', default=8, type=int, help='snn simulation time')

# model configuration
parser.add_argument('-data', '--dataset',default='imagenet',type=str,help='dataset')    #imagenet, cifar10, cifar100
parser.add_argument('-arch','--model',default='resnet20',type=str,help='model')    #resnet18，resnet20，resnet34，vgg16 

# training configuration
parser.add_argument('--epochs',default=300,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.05,type=float,metavar='LR', help='initial learning rate') # 0.05 for cifar100 / 0.1 for cifar10
parser.add_argument('-wd','--weight_decay',default=5e-4, type=float, help='weight_decay')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-L', '--L', default=4, type=int, help='Step L')

# train_with_thre 相关参数
parser.add_argument('--use_thre_training', action='store_true', help='使用阈值自适应训练')
parser.add_argument('--adjust_interval', default=10, type=int, help='阈值调整间隔')
parser.add_argument('--probe_num_batches', default=3, type=int, help='梯度探测批次数')
parser.add_argument('--adjust_scale_factor', default=0.01, type=float, help='层权重调整缩放因子')
parser.add_argument('--return_thre_grads', action='store_true', help='返回阈值梯度历史')
parser.add_argument('--scale_factor', default=5.0, type=float, help='阈值更新中的缩放因子')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size)
    # preparing model
    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)
    model.set_T(args.time)     # Cray 设置时间步长

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0
    
    # 初始化层权重（用于阈值自适应训练）
    layer_weights = None
    if args.use_thre_training:
        # 统计IF层数量
        if_layers = [m for m in model.modules() if hasattr(m, 'thresh') and m.thresh is not None]
        if len(if_layers) > 0:
            layer_weights = torch.ones(len(if_layers), device=device) * 0.01  # 初始化为小值
            print(f"初始化 {len(if_layers)} 个IF层的权重: {layer_weights}")
        else:
            print("警告: 未找到IF层，将使用标准训练")
            args.use_thre_training = False

    identifier = args.model

    identifier += '_L[%d]'%(args.L)
    
    # 添加训练模式标识
    if args.use_thre_training:
        identifier += '_thre[%d,%d,%.3f]'%(args.adjust_interval, args.probe_num_batches, args.adjust_scale_factor)

    if not args.suffix == '':
        identifier += '_%s'%(args.suffix)

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    
    # 记录训练模式
    if args.use_thre_training:
        logger.info('使用阈值自适应训练模式')
        logger.info(f'调整间隔: {args.adjust_interval}, 探测批次数: {args.probe_num_batches}')
        logger.info(f'调整缩放因子: {args.adjust_scale_factor}')
    else:
        logger.info('使用标准训练模式')
    
    for epoch in range(args.epochs):
        if args.use_thre_training:
            # 使用阈值自适应训练
            loss, acc = train_with_thre(
                model, device, train_loader, criterion, optimizer, args.time,
                scale_factor=args.scale_factor,
                layer_weights=layer_weights,
                adjust_interval=args.adjust_interval,
                return_thre_grads=args.return_thre_grads,
                probe_num_batches=args.probe_num_batches,
                probe_loader=None,
                adjust_scale_factor=args.adjust_scale_factor
            )
        else:
            # 使用标准训练
            loss, acc = train(model, device, train_loader, criterion, optimizer, args.time)
        
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
