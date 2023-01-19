import os
import sys
import yaml
import torch
import argparse

import trainer
from utils import metrics
from model import *
from dataset import *

from torch.utils.data import DataLoader


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # 1. 获取参数配置
    train_config = load_config(args.model_config_path)

    # model_config =

    # 2. dataset

    # scalar

    # 3. model = getattr model-config

    # 4. dataloader: model-dataset

    # 5. trainer

if __name__ == '__main__':
    print('111')

    # 定义解析器
    parser = argparse.ArgumentParser()
    # 加参数
    parser.add_argument('--model_config_path', type=str, default='./config/training_config.yaml')


    # 命令行取参数
    args = parser.parse_args()
    main(args)
