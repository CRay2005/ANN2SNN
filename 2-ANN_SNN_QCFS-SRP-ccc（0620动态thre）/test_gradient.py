import argparse
import os
import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from layer import GradientAnalyzer

model = 
def main():
    analyzer = GradientAnalyzer(model, prune_ratio=0.1)
    for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()  # 反向传播会触发梯度钩子
    optimizer.step()
    low_grad_neurons = analyzer.get_low_grad_neurons()