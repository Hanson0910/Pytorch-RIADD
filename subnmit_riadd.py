import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import numpy as np

from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
# from timm.data import LoadImagesAndLabels,preprocess,LoadImagesAndLabelsV2,LoadImagesAndSoftLabels
from timm.utils import ApexScaler,get_score
from timm.utils import Visualizer
from timm.data import get_riadd_train_transforms,get_riadd_valid_transforms,get_riadd_test_transforms
from timm.data import RiaddDataSet,RiaddDataSet9Classes

import os
from tqdm import tqdm
import random
import torch.distributed as dist
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

CFG = {
    'seed': 42,
    'img_size': 960,
    'valid_bs': 20,
    'num_workers': 4,
    'num_classes': 29,
    'tta': 3,
    'models':['tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V2-8Classes-SGD1E-1fold_0/train/20210225-225117-tf_efficientnet_b6_ns-768/model_best.pth.tar',
              'tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V2-8Classes-SGD1E-1fold_1/train/20210226-001105-tf_efficientnet_b6_ns-768/model_best.pth.tar',
              'tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V2-8Classes-SGD1E-1fold_2/train/20210226-013350-tf_efficientnet_b6_ns-768/model_best.pth.tar',
              'tf_efficientnet_b6_ns-RIADD-DDP-RemoveBlack-768-V2-8Classes-SGD1E-1fold_3/train/20210226-025923-tf_efficientnet_b6_ns-768/model_best.pth.tar'],
    'base_img_path': '/media/ExtDiskB/Hanson/datasets/wheat/RIADD/valid',
    'weights': [1,1,1,1]
}



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def validate(model, loader): 
    model.eval()
    preds = []
    pbar = tqdm(enumerate(loader), total=len(loader))  
    with torch.no_grad():
        for batch_idx, (input, target) in pbar:
            input = input.cuda()
            target = target.cuda()
            target = target.float()
            output = model(input)
            preds.append(output.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

if __name__ == '__main__':
    from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold
    import pandas as pd
    import torch.utils.data as data
    seed_everything(CFG['seed'])
    data_ = pd.read_csv('/media/ExtDiskB/Hanson/datasets/wheat/RIADD/teamName_results.csv')
    test_index = [i for i in range(data_.shape[0])]

    test_data = data_.iloc[test_index, :].reset_index(drop=True) 
    test_transforms = get_riadd_test_transforms(CFG)
    test_dataset = RiaddDataSet9Classes(image_ids = test_data,transform = test_transforms, 
                                baseImgPath = CFG['base_img_path'])
    test_data_loader = data.DataLoader( test_dataset, 
                                        batch_size=CFG['valid_bs'], 
                                        shuffle=False, 
                                        num_workers=CFG['num_workers'], 
                                        pin_memory=True, 
                                        drop_last=False,
                                        sampler = None)
    
    imgIds = test_data.iloc[test_index,0].tolist()
    target_cols = test_data.iloc[test_index, 1:].columns.tolist()    
    test = pd.DataFrame()
    test['ID'] = imgIds

    tst_preds = []
    for i,model_name in enumerate(CFG['models']):
        if model_name.find('ResNet200D') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'resnet200d',num_classes=CFG['num_classes'])
        
        if model_name.find('nf_resnet50') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'nf_resnet50',num_classes=CFG['num_classes'])
        
        if model_name.find('tf_efficientnet_b7_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b7_ns',num_classes=CFG['num_classes'])
        
        if model_name.find('tf_efficientnet_b4_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b4_ns',num_classes=CFG['num_classes'])

        if model_name.find('tf_efficientnet_b6_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b6_ns',num_classes=CFG['num_classes'])

        if model_name.find('tf_efficientnet_b5_ns') != -1:
            model_path = os.path.join('/media/ExtDiskB/Hanson/code/RANZCR/pytorch-image-models-master/ckpt',model_name)
            model = create_model(model_name = 'tf_efficientnet_b5_ns',num_classes=CFG['num_classes'])
        
        print('model_path: ',model_path)
        state_dict = torch.load(model_path,map_location='cpu')
        model.load_state_dict(state_dict["state_dict"], strict=True)
        model = nn.DataParallel(model)
        model.cuda()
        for _ in range(CFG['tta']):
            tst_preds += [CFG['weights'][i]/sum(CFG['weights'])/CFG['tta']*validate(model,test_data_loader)]
    tst_preds = np.sum(tst_preds, axis=0)
    test[target_cols] = tst_preds
    test.to_csv('submission_eb6-768-29classes.csv', index=False)
