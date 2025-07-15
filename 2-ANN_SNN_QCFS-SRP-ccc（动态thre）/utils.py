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

import os
import datetime

def val(model, test_loader, device, T,optimize_thre_flag=False):
    correct = 0
    total = 0
    model.eval()

    for m in model.modules():
        if isinstance(m, IF):
            m.optimize_thre_flag = optimize_thre_flag

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

            # # 采集最优thre，运行指定batchsize就退出
            # if optimize_thre_flag and batch_idx == 1:
            #     # 保存每个IF层的阈值数据
            #     if_count = 0  # 用于记录IF层的索引
            #     for m in model.modules():
            #         if isinstance(m, IF):
            #             os.makedirs('log', exist_ok=True)
            #             # 为每个IF层创建独立的CSV文件
            #             save_path = os.path.join('log', f'IF_{if_count}_thresholds.csv')
            #             m.save_thresholds_to_csv(save_path)
            #             if_count += 1
            #     print("保存每个IF层的阈值数据完成！")
            #     # print(f"batch:{batch_idx}")
            #     # print(f"final_acc:{100 * correct / total}")
            #     break

        final_acc = 100 * correct / total
    return final_acc
