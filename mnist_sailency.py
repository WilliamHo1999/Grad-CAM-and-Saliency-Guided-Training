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

from utils.Data import PascalVOC, dataset_voc, ImageNet300Dataset, MNIST
from GradCam import GradCam
from utils.SGT import Saliency_loss, Masker
from utils.Models import SimpleCNN, DeeperCNN
from utils.Trainer import Trainer

import matplotlib.pyplot as plt


MNIST_ROOT_DIR = ''

def seed_everything(seed_value=2021):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mnist_normal():
    device = 'cuda'

    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(20),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
                        ),
      'val': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
        )
    }

    image_datasets={}
    image_datasets['train']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=True, transform = data_transforms['train'])
    image_datasets['val']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=False, transform = data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=1)

    classes = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    lr = 0.001
    multi_label = False

    model = DeeperCNN()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                optimizer,
                                milestones=[10, 25, 30, 60, 90, 120],
                                gamma=0.3,
                            )
    
    loss_fn = nn.CrossEntropyLoss()


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
        num_epochs = 30,
        multi_label = multi_label,
        model_name = f'model_name',
        model_params = model_params,
        full_train = True,
        sailency = False,
    )


    start = time.time()
    trainer.train()
    end = time.time() - start


    current_results = {
        'model_name':trainer.model_name,
        'learning_rate':lr,
        'best_eval_acc':trainer.best_measure,
        'train_acc_at_best':trainer.train_measure_at_best,
        'best_epoch':trainer.best_epoch,
        'multi_label':multi_label,
        'time':end,
        'total_epochs':trainer.latest_epoch,
        'avg_time_per_epoch':end/trainer.latest_epoch
    }

    results = results.append(current_results, ignore_index = True)

    results.sort_values("best_eval_acc").to_csv(f'train_fashion_mnist_normal.csv')



def mnist_compare_sailency():

    device = 'cuda'
    torch.cuda.empty_cache()


    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(20),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
                        ),
      'val': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
        )
    }

    image_datasets={}
    image_datasets['train']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=True, transform = data_transforms['train'])
    image_datasets['val']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=False, transform = data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=1)

    classes = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    k_upper = 50
    lr = 0.001
    multi_label = False

    for simple_model in [False]:
        for sail_train in [True, False]:

            if simple_model:
                model = SimpleCNN()
            else:
                model = DeeperCNN()
            model.to(device)
            print(model)

            optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                        optimizer,
                                        milestones=[10, 25, 30, 60, 90, 120],
                                        gamma=0.3,
                                    )
            
            loss_fn = nn.CrossEntropyLoss()


            if sail_train:
                mask_optim = torch.optim.AdamW(model.parameters(), lr = 0.005)
                masker = Masker(model, mask_optim, k_upper = k_upper, multi_label = multi_label)
                loss_fn = Saliency_loss(loss_fn, 1, multi_label = multi_label)#.to(device)
            else:
                masker = None

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
                num_epochs = 30,
                multi_label = multi_label,
                model_name = f'make_a_name',
                model_params = model_params,
                full_train = False,
                sailency = True,
                masker = masker,
            )


            start = time.time()
            trainer.train()
            end = time.time() - start


            current_results = {
                'model_name':trainer.model_name,
                'simple_model':str(simple_model),
                'sailancy_train':str(True),
                'pixels_masked': k_upper,
                'learning_rate':lr,
                'best_eval_acc':trainer.best_measure,
                'train_acc_at_best':trainer.train_measure_at_best,
                'best_epoch':trainer.best_epoch,
                'multi_label':multi_label,
                'time':end,
                'total_epochs':trainer.latest_epoch,
                'avg_time_per_epoch':end/trainer.latest_epoch
            }

            results = results.append(current_results, ignore_index = True)

            results.sort_values("best_eval_acc").to_csv(f'train_mnist.csv')
    results.sort_values("best_eval_acc").to_csv(f'train_mnist.csv')


def mask_tuning():  

    device = 'cuda'
    torch.cuda.empty_cache()

    results = pd.DataFrame()

    data_transforms = {
      'train': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomCrop(20),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
                        ),
      'val': transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.1307,), (0.3081,),)
                             ]
        )
    }

    image_datasets={}
    image_datasets['train']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=True, transform = data_transforms['train'])
    image_datasets['val']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest=False, transform = data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=1)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=1)

    classes = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    k_upper = 50
    lr = 0.001
    multi_label = False
    sail_train = True

    for simple_model in [False]:
        for absolute_min in [True, False]:
            for k_upper in [300,500,int((28*28)/2), 50, 150, 250, 300, 350, 500]:
                print(simple_model, absolute_min, k_upper)

                if simple_model:
                    model = SimpleCNN()
                else:
                    model = DeeperCNN()
                model.to(device)
                print(model)

                optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                            optimizer,
                                            milestones=[8, 25, 30, 60, 90, 120],
                                            gamma=0.3,
                                        )
                
                loss_fn = nn.CrossEntropyLoss()


                if sail_train:
                    mask_optim = torch.optim.AdamW(model.parameters(), lr = 0.005)
                    masker = Masker(model, mask_optim, k_upper = k_upper, absolute_min = absolute_min, multi_label = multi_label)
                    loss_fn = Saliency_loss(loss_fn, 1, multi_label = multi_label)#.to(device)
                else:
                    masker = None

                model_params = {
                    'optimizer':optimizer,
                    'scheduler':scheduler,
                    'data_loader':dataloaders
                }


                trainer = Trainer(
                    model = model,
                    loss_fn = loss_fn,
                    classes = classes,
                    saliency_guided_training = sail_train,
                    num_epochs = 30,
                    multi_label = multi_label,
                    model_name = f'make_a_name',
                    model_params = model_params,
                    full_train = True,
                    sailency = sail_train,
                    masker = masker,
                )


                start = time.time()
                trainer.train()
                end = time.time() - start


                current_results = {
                    'model_name':trainer.model_name,
                    'simple_model':str(simple_model),
                    'sailancy_train':str(sail_train),
                    'pixels_masked': k_upper,
                    'absolute_min': str(absolute_min), 
                    'learning_rate':lr,
                    'best_eval_acc':trainer.best_measure,
                    'train_acc_at_best':trainer.train_measure_at_best,
                    'best_epoch':trainer.best_epoch,
                    'multi_label':multi_label,
                    'time':end,
                    'total_epochs':trainer.latest_epoch,
                    'avg_time_per_epoch':end/trainer.latest_epoch
                }

                results = results.append(current_results, ignore_index = True)

                results.sort_values("best_eval_acc").to_csv(f'a.csv')
    results.sort_values("best_eval_acc").to_csv(f'a.csv')


if __name__ == '__main__':
    seed_everything(2021)
    #mask_tuning()
    #mnist_compare_sailency()
    #train_mnist_normal()