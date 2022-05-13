import os
import cv2
import numpy as np
import sklearn.metrics as metrics
import random
import PIL.Image
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils

from utils.Data import PascalVOC, dataset_voc, ImageNet300Dataset, MNIST_MultiLabel
from GradCam import GradCam
from utils.SGT import Saliency_loss, Masker
from utils.Models import SimpleCNN, DeeperCNN
from utils.Trainer import Trainer

import matplotlib.pyplot as plt


ROOT_DIR_PASCAL_VOC = 'data_dir'

def seed_everything(seed_value=2021):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_pascal_voc_normal_single():

    device = 'cuda'
    torch.cuda.empty_cache()
    torch.manual_seed(2021)
    random.seed(2021)
    np.random.seed(2021)

    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize((256,256)),
          #transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    image_datasets={}
    image_datasets['train']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=0, transform = data_transforms['train'])

    image_datasets['val']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=1, transform = data_transforms['val'])


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=1)

    classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    k_upper = 50
    lr = 0.005
    fine_tune = False
    multi_label = True

    model = models.resnet18(pretrained=True)
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, 20)
    model.fc.reset_parameters()
    model.to(device)
    print(model)

    
    if fine_tune:
        model_params = [param for name, param in model.named_parameters() if 'fc' not in name]
        optimizer = torch.optim.AdamW([ {'params': model_params},
                                        {'params':model.fc.parameters(), 'lr': lr}]
                                        , lr = 0.0001
                                        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer,
                                    milestones=[5, 13, 18, 20, 25],
                                    gamma=0.3,
                                )

    else:
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer,
                                    milestones=[5, 13, 18, 20, 25],
                                    gamma=0.3,
                                )
    print(optimizer)
    loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')#.to(device)

    model_params = {
        'optimizer':optimizer,
        'scheduler':scheduler,
        'data_loader':dataloaders
    }


    trainer = Trainer(
        model = model,
        loss_fn = loss_fn,
        classes = classes,
        saliency_guided_training = False,
        num_epochs = 20,
        model_name = f'model_name',
        model_params = model_params,
        multi_label = multi_label,
        full_train = True,
        sailency = False,
    )

    start = time.time()
    trainer.train()
    end = time.time() - start


    current_results = {
        'model_name':trainer.model_name,
        'fine_tune':str(fine_tune),
        'learning_rate':lr,
        'best_eval_mape':trainer.best_measure,
        'train_acc_at_best':trainer.train_measure_at_best,
        'best_epoch':trainer.best_epoch,
        'multi_label':multi_label,
        'time':end,
        'total_epochs':trainer.latest_epoch,
        'avg_time_per_epoch':end/trainer.latest_epoch
    }

    results = results.append(current_results, ignore_index = True)

    results.sort_values("best_eval_mape").to_csv(f'model_name.csv')




def train_pascal_voc_normal():

    device = 'cuda'
    torch.cuda.empty_cache()

    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize((256,256)),
          #transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    image_datasets={}
    image_datasets['train']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=0, transform = data_transforms['train'])

    image_datasets['val']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=1, transform = data_transforms['val'])


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=1)

    classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    k_upper = 50
    lr = 0.001
    multi_label = True

    for ok in [False]:
        for fine_tune in [True, False]:
            for lr in [0.005, 0.001]:
                print(fine_tune, ok, lr)

                model = models.resnet18(pretrained=True)
                num_ft = model.fc.in_features
                model.fc = nn.Linear(num_ft, 20)
                model.fc.reset_parameters()
                model.to(device)
                print(model)

                
                if fine_tune:
                    model_params = [param for name, param in model.named_parameters() if 'fc' not in name]
                    optimizer = torch.optim.AdamW([ {'params': model_params},
                                                    {'params':model.fc.parameters(), 'lr': lr}]
                                                    , lr = 0.0001
                                                    )

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                optimizer,
                                                milestones=[5, 10, 15, 20, 25],
                                                gamma=0.3,
                                            )

                else:
                    optimizer = torch.optim.AdamW(model.fc.parameters(), lr)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                optimizer,
                                                milestones=[5, 10, 15, 20, 25],
                                                gamma=0.3,
                                            )
                print(optimizer)
                loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')#.to(device)

                model_params = {
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'data_loader':dataloaders
                }


                trainer = Trainer(
                    model = model,
                    loss_fn = loss_fn,
                    classes = classes,
                    saliency_guided_training = False,
                    num_epochs = 20,
                    model_name = f'model_name_pick_a_name',
                    model_params = model_params,
                    multi_label = multi_label,
                    full_train = False,
                    sailency = False,
                )

                start = time.time()
                trainer.train()
                end = time.time() - start


                current_results = {
                    'model_name':trainer.model_name,
                    'fine_tune':str(fine_tune),
                    'sailancy_train':str(False),
                    'learning_rate':lr,
                    'best_eval_mape':trainer.best_measure,
                    'train_acc_at_best':trainer.train_measure_at_best,
                    'best_epoch':trainer.best_epoch,
                    'multi_label':multi_label,
                    'time':end,
                    'total_epochs':trainer.latest_epoch,
                    'avg_time_per_epoch':end/trainer.latest_epoch
                }

                results = results.append(current_results, ignore_index = True)

                results.sort_values("best_eval_mape").to_csv(f'pascal_voc_normal.csv')

    results.sort_values("best_eval_mape").to_csv(f'pascal_voc_normal.csv')


        
def mask_tuning():
    device = 'cuda'
    torch.cuda.empty_cache()


    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize((256,256)),
          #transforms.CenterCrop(256),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }

    image_datasets={}
    image_datasets['train']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=0, transform = data_transforms['train'])

    image_datasets['val']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=1, transform = data_transforms['val'])


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=1)

    classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    k_upper = 50
    lr = 0.005
    multi_label = True

    # Wrong, but whatever... still masks too many features
    num_features = 224*224*3

    k_list = [int(num_features/2), int(num_features/4), int(num_features/10), int(num_features/100)]

    
    for absolute_min in [True, False]:
        for fine_tune in [True,False]:
            for k_upper in k_list:
                print(fine_tune, absolute_min, k_upper)

                model = models.resnet18(pretrained=True)
                num_ft = model.fc.in_features
                model.fc = nn.Linear(num_ft, 20)
                model.fc.reset_parameters()
                model.to(device)
                print(model)

                
                if fine_tune:
                    model_params = [param for name, param in model.named_parameters() if 'fc' not in name]
                    optimizer = torch.optim.AdamW([ {'params': model_params},
                                                    {'params':model.fc.parameters(), 'lr': lr}]
                                                    , lr = 0.0001
                                                    )

                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                optimizer,
                                                milestones=[5, 10, 15, 20, 25],
                                                gamma=0.3,
                                            )

                else:
                    optimizer = torch.optim.AdamW(model.fc.parameters(), lr)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                optimizer,
                                                milestones=[5, 10, 15, 20, 25],
                                                gamma=0.3,
                                            )
                print(optimizer)
                loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction='mean')#.to(device)

                mask_optim = torch.optim.AdamW(model.parameters(), lr = 0.005)
                masker = Masker(model, mask_optim, k_upper = k_upper, absolute_min=absolute_min, multi_label = multi_label)
                loss_fn = Saliency_loss(loss_fn, 1, multi_label = multi_label)#.to(device)

                model_params = {
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'data_loader':dataloaders
                }


                trainer = Trainer(
                    model = model,
                    loss_fn = loss_fn,
                    classes = classes,
                    saliency_guided_training = True,
                    num_epochs = 20,
                    model_name = f'model_name',
                    model_params = model_params,
                    multi_label = multi_label,
                    full_train = True,
                    sailency = True,
                    masker = masker,
                )

                start = time.time()
                trainer.train()
                end = time.time() - start


                current_results = {
                    'model_name':trainer.model_name,
                    'fine_tune':str(fine_tune),
                    'sailancy_train':str(True),
                    'absolute_min': str(absolute_min),
                    'pixels_masked': k_upper,
                    'learning_rate':lr,
                    'best_eval_mape':trainer.best_measure,
                    'train_acc_at_best':trainer.train_measure_at_best,
                    'best_epoch':trainer.best_epoch,
                    'multi_label':multi_label,
                    'time':end,
                    'total_epochs':trainer.latest_epoch,
                    'avg_time_per_epoch':end/trainer.latest_epoch
                }

                results = results.append(current_results, ignore_index = True)

                results.sort_values("best_eval_mape").to_csv(f'c.csv')

    results.sort_values("best_eval_mape").to_csv(f'c.csv')

    


if __name__ == '__main__':
    seed_everything(2021)
    #train_pascal_voc_normal_single()
    #mask_tuning()
    #train_pascal_voc_normal()