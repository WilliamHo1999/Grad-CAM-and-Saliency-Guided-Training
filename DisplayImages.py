import cv2
import numpy as np
import sklearn.metrics as metrics
import random
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils

from utils.Data import PascalVOC, dataset_voc, ImageNet300Dataset, MNIST, MNIST_MultiLabel
from utils.getimagenetclasses import get_classes
from utils.ImageDisplayer import ImageDisplayer
from GradCam import GradCam
from utils.Models import DeeperCNN


import matplotlib.pyplot as plt


ROOT_DIR_PASCAL_VOC = ''
ROOT_DIR_IMAGENET300 = ''
XML_LABEL_DIR = ''
SYSNET_FILE = ''
MNIST_ROOT_DIR = ''


def seed_everything(seed_value=2021):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def display_pascal_voc():
    device = 'cuda'
    torch.cuda.empty_cache()
    
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
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }


    
    image_datasets={}
    image_datasets['val']= dataset_voc(root_dir=ROOT_DIR_PASCAL_VOC, trvaltest=1, transform = data_transforms['val'])


    dataloaders = {}
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=1)


    model = models.resnet18(pretrained=True)
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, 20)
    model.fc.reset_parameters()
    model.load_state_dict(torch.load('model_path'))
    model.to(device)
    print(model)
    model.eval()

    target_layer = model.layer4[-1].conv2

    cam = GradCam(model, target_layer,10)

    classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']


    image_dispalyer = ImageDisplayer(model, 
        cam, 
        classes,
        reshape = transforms.Resize((256,256)), 
        multi_label = True, 
        image_dir = 'pascal_dir',
        pdf = False)

    for i, img in enumerate(dataloaders['val']):

        img_1 = img
        img_1['image'] = img_1['image'].squeeze(0)
        img_1['label'] = img_1['label'].squeeze(0)
        img_1['filename'] = img_1['filename'][0]


        image_dispalyer.display_images(img_1, display_labels_or_predictions = False)
        image_dispalyer.display_images(img_1, display_labels_or_predictions = True)



def display_image_net():

    device = 'cuda'
    torch.cuda.empty_cache()

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
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    }


    image_datasets={}
    image_datasets['val']= ImageNet300Dataset(root_dir=ROOT_DIR_IMAGENET300, xmllabeldir = XML_LABEL_DIR, synsetfile = SYSNET_FILE, maxnum=300, transform = data_transforms['val'])
    dataloaders = {}
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=1)


    model = models.resnet18(pretrained=True)
    model.to(device)

    model.eval()

    target_layer = model.layer4[-1].conv2

    classes = get_classes()

    cam = GradCam(model, target_layer,10, multi_label = False)

    image_dispalyer = ImageDisplayer(model, 
        cam, 
        classes,
        reshape = transforms.Resize((256,256)), 
        multi_label = False, 
        image_dir = 'image_net_dir',
        pdf = False)

    
    for i in range(len(dataloaders['val'].dataset)):
        img_1 = dataloaders['val'].dataset[i]

        image_dispalyer.display_images(img_1, display_labels_or_predictions = True)
        image_dispalyer.display_images(img_1, display_labels_or_predictions = False)
    

def display_mnist():
    device = 'cuda'
    torch.cuda.empty_cache()

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
                             ]
        )
    }


    image_datasets={}
    image_datasets['val']= MNIST(root_dir=MNIST_ROOT_DIR, trvaltest = False, transform = data_transforms['val'])
    dataloaders = {}
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=1)


    model = DeeperCNN()
    model.load_state_dict(torch.load('model_path'))
    model.to(device)
    print(model)

    model.eval()

    target_layer = model.cnn6

    classes =  ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    cam = GradCam(model, target_layer, 10, multi_label = False)
    image_dispalyer = ImageDisplayer(model, 
        cam, 
        classes, 
        multi_label = False, 
        mnist = True,
        image_dir = 'mnist_examples', pdf = True)
    cam_2 = GradCam(model, target_layer, 10, multi_label = False, no_relu = True)
    image_dispalyer_2 = ImageDisplayer(model, 
        cam_2, 
        classes, 
        multi_label = False, 
        mnist = True,
        image_dir = 'mnist_examples', pdf = True)
    cam_3 = GradCam(model, target_layer, 10, multi_label = False, negative_activation = True)
    image_dispalyer_3 = ImageDisplayer(model, 
        cam_3, 
        classes, 
        multi_label = False, 
        mnist = True,
        image_dir = 'mnist_examples', pdf = True)

    
    print(len(dataloaders['val'].dataset))
    for i in range(10,20):
        img_1 = dataloaders['val'].dataset[i]

        image_dispalyer.display_images(img_1, display_labels_or_predictions = True, file_name = f'mnist_{i}')
        image_dispalyer_2.display_images(img_1, display_labels_or_predictions = True, file_name = f'mnist_{i}_no_relu')
        image_dispalyer_3.display_images(img_1, display_labels_or_predictions = True, file_name = f'mnist_{i}_negative_activation')



def display_mnist_multi_label():
    device = 'cuda'
    torch.cuda.empty_cache()

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
    image_datasets['val']= MNIST_MultiLabel(root_dir=MNIST_ROOT_DIR, trvaltest = False, transform = data_transforms['val'])
    dataloaders = {}
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=1)


    model = DeeperCNN(True)
    model.load_state_dict(torch.load('path_name'))
    model.to(device)
    print(model)

    model.eval()

    target_layer = model.cnn6

    classes = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    cam = GradCam(model, target_layer, 10, multi_label = True)

    image_dispalyer = ImageDisplayer(model, 
        cam, 
        classes, 
        multi_label = True, 
        mnist = True,
        image_dir = 'mnist_multi_label_examples', 
        pdf = True
    )
    cam_2 = GradCam(model, target_layer, 10, multi_label = True, no_relu = True)
    image_dispalyer_2 = ImageDisplayer(model, 
        cam_2, 
        classes, 
        multi_label = True, 
        mnist = True,
        image_dir = 'mnist_multi_label_examples', 
        pdf = True
    )
    cam_3 = GradCam(model, target_layer, 10, multi_label = True, negative_activation = True)
    image_dispalyer_3 = ImageDisplayer(model, 
        cam_3, 
        classes, 
        multi_label = True, 
        mnist = True,
        image_dir = 'mnist_multi_label_examples', 
        pdf = True
    )
    print(len(dataloaders['val'].dataset))
    
    for i in dataloaders['val']:

        img_1 = dataloaders['val'].dataset[i]

        image_dispalyer.display_images(img_1, file_name = f'mnist_multi_label_{i}', display_labels_or_predictions = True)
        image_dispalyer_2.display_images(img_1, display_labels_or_predictions = True, file_name = f'mnist_multi_label_{i}_no_relu')
        image_dispalyer_3.display_images(img_1, display_labels_or_predictions = True, file_name = f'mnist_multi_label_{i}_negative_activation')


if __name__ == "__main__":
    seed_everything()
    
    #display_mnist_multi_label()
    #display_mnist()
    #display_pascal_voc()
    #display_image_net()