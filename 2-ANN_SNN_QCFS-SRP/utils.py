import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from Models import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total
'''
def val(model, test_loader, device, T):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            inputs = inputs.to(device)
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc

'''

import os
import datetime
from torchsummary import summary # 用于预览神经网络结构的库

def val(model, test_loader, device, T):
    correct = 0
    total = 0
    model.eval()

    # cxqian test version
    # 定义钩子函数
    def save_hook(m, x, y):
        #print(f"Input tensor:\n{x[0]}")     
        #print(f"Outnput tensor:\n{y}")   
        print(f"m.thresh:{m.thresh}")
        # return
        # 获取当前时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建保存目录
        os.makedirs("hook_outputs", exist_ok=True)
        # 生成文件名
        filename = f"hook_outputs/{m.__class__.__name__}_{timestamp}.txt"
        # 保存输出到文件
        #torch.save(x[0], filename)
        #torch.save(y, filename)

        with open(filename, 'a') as f:            
            f.write(f"Input:{x[0].shape[1]}\n")
            np.savetxt(f, x[0].detach().cpu().numpy().reshape(-1, x[0].shape[1]), fmt='%f')
            f.write(f"Output:{y.shape[1]}\n")
            np.savetxt(f, y.detach().cpu().numpy().reshape(-1, y.shape[1]), fmt='%f')
            f.write("\n")  # 添加换行以分隔不同的输入输出对
        print(f"Saved output of {m.__class__.__name__} to {filename}")
        return
    
    '''
    以(layer1)为例，
    输入:3,  32 * 32  * 200(batchsize)  *8(T) =1638400
    输出:64, 16 * 16   ????
    '''  

    #查找IF层，返回到一个列表中
    '''
    #用递归方式实现有问题
    def list_modules(model):  
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                list_modules(module)
            print(f"module name:{name}\n")
            print(f"__class__.__name__:{module.__class__.__name__}\n")
            if 'IF' in module.__class__.__name__:
                print(f"return_layer:{module}\n")
                IF_model=module
                break    
        return IF_model
    '''
    def list_modules(model):  
        NFmodules=[]
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                for sub_name, sub_module in module._modules.items():
                    #print(f"sub_module name:{sub_name}\n")
                    #print(f"__class__.__name__:{sub_module.__class__.__name__}\n")
                    if 'IF' in sub_module.__class__.__name__:
                        #print(f"return_layer:{sub_module}\n")
                        NFmodules.append(sub_module)   
            if 'IF' in module.__class__.__name__:
                #print(f"return_layer:{module}\n")
                NFmodules.append(module)              
        return NFmodules


    #对于给定的输入size，查看整个网络模型的输入输出结构
    #summary(model, input_size=[[3, 32, 32]])
    
    # test_layer = list_modules(model)
    
    # 打印模型网络结构
    # print("="*50)
    # print("模型网络结构:")
    # print("="*50)
    # print(model)
    # print("="*50)
    # print("模型各层详细信息:")
    # print("="*50)
    # for name, module in model.named_modules():
    #     print(f"层名: {name}, 层类型: {module.__class__.__name__}")
    #     if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
    #         print(f"  输入特征数: {module.in_features}, 输出特征数: {module.out_features}")
    #     elif hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
    #         print(f"  输入通道数: {module.in_channels}, 输出通道数: {module.out_channels}")
    #         if hasattr(module, 'kernel_size'):
    #             print(f"  卷积核大小: {module.kernel_size}")
    #     if 'IF' in module.__class__.__name__:
    #         print(f"  IF层参数 - T: {module.T}, 阈值: {module.thresh.data}, tau: {module.tau}")
    # print("="*50)
    
    #print(f"NFmodels:{test_layer}\n")
    
    #print(f"test_layer:{test_layer[6]}\n")
    #print("注册钩子！")
    #hook = test_layer[0].register_forward_hook(save_hook)

    

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((test_loader)):
            inputs = inputs.to(device)
            if T > 0:
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            # # 为了便于测试，仅运行一个batchsize就退出
            # if batch_idx == 10:
            #     break
            # print(f"batch:{batch_idx}\n")
            # print(f"final_acc:{100 * correct / total}\n")
        final_acc = 100 * correct / total
    return final_acc
