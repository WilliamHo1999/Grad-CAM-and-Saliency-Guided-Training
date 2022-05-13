import os

import numpy as np
import PIL.Image
import pandas as pd

import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from utils.getimagenetclasses import get_classes, parsesynsetwords, parseclasslabel


class MNIST(Dataset):
    def __init__(self, root_dir, trvaltest, fashionMNIST = True, transform=None):
        
        self.fashionMNIST = fashionMNIST

        if not self.fashionMNIST:
            self.data = torchvision.datasets.MNIST(root = root_dir, train = trvaltest, transform = transform)
        else:
            self.data = torchvision.datasets.FashionMNIST(root = root_dir, download = True, train = trvaltest, transform = transform)


        self.images = self.data.data
        self.labels = self.data.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        images = self.images[idx].type(torch.FloatTensor).unsqueeze(0)
        return {'image': images, 'label':self.labels[idx]}


class MNIST_MultiLabel(Dataset):
    def __init__(self, root_dir, trvaltest, fashionMNIST = True, transform=None):
        
        self.fashionMNIST = fashionMNIST

        if not self.fashionMNIST:
            self.data = torchvision.datasets.MNIST(root = root_dir, train = trvaltest, transform = transform)
        else:
            self.data = torchvision.datasets.FashionMNIST(root = root_dir, download = True, train = trvaltest, transform = transform)

        self.images = self.data.data
        self.labels = self.data.targets

        self.rest = (len(self.data) % 4)
        self.num_samples = len(self.data) - self.rest

        self.combined_images = []
        self.combined_labels = torch.zeros((self.num_samples, 10))

        start = 0
        for sample_num, i in enumerate(range(4, self.num_samples, 4)):
            current_images = self.images[start:i]
            current_labels = self.labels[start:i]
            
            top_image = torch.cat((current_images[0], current_images[1]))
            bot_image = torch.cat((current_images[2], current_images[3]))
            full_image = torch.cat((top_image, bot_image), axis = 1)

            label = torch.zeros(10)
            label[current_labels] = 1

            self.combined_labels[sample_num, :] = label

            self.combined_images.append(full_image)
            
            start = i

    def __len__(self):
        return len(self.combined_images)

    def __getitem__(self,idx):
        
        images = self.combined_images[idx].type(torch.FloatTensor).unsqueeze(0)
        
        return {'image': images, 'label':self.combined_labels[idx]}


class dataset_voc(Dataset):
    def __init__(self, root_dir, trvaltest, transform=None):

        self.root_dir = root_dir

        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]

        pv=PascalVOC(root_dir)
        cls=pv.list_image_sets()

        if trvaltest==0:
            dataset='train'
        elif trvaltest==1:
            dataset='val'
        else:
            print('Not a split')
            exit()

        
        filenamedict={}
        for c,cat_name in enumerate(cls):
            imgstubs=pv.imgs_from_category_as_list(cat_name, dataset)
            for st in imgstubs:
                if st in filenamedict:
                    filenamedict[st][c]=1
                else:
                    vals=np.zeros(20,dtype=np.int32)
                    vals[c]=1
                    filenamedict[st]=vals


        self.labels=np.zeros((len(filenamedict),20))
        tmpct=-1
        for key,value in filenamedict.items():
            tmpct+=1
            self.labels[tmpct,:]=value

            fn=os.path.join(self.root_dir,'JPEGImages',key+'.jpg')
            self.imgfilenames.append(fn)


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
        label = self.labels[idx,:].astype(np.float32)

        if self.transform:
            image = self.transform(image)

        if image.size()[0]==1:
            image=image.repeat([3,1,1])

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample


class PascalVOC:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values



class ImageNet300Dataset(Dataset):
    def __init__(self, root_dir, xmllabeldir, synsetfile, maxnum, transform=None):

        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.xmllabeldir=xmllabeldir
        self.transform = transform
        self.imgfilenames=[]
        self.labels=[]
        self.ending=".JPEG"

        self.clsdict=get_classes()


        indicestosynsets,self.synsetstoindices,synsetstoclassdescr=parsesynsetwords(synsetfile)


        for root, dirs, files in os.walk(self.root_dir):
            for ct,name in enumerate(files):
                nm=os.path.join(root, name)
                #print(nm)
                if (maxnum >0) and ct>= (maxnum):
                    break
                self.imgfilenames.append(nm)
                label,firstname=parseclasslabel(self.filenametoxml(nm) ,self.synsetstoindices)
                self.labels.append(label)

   
    def filenametoxml(self,fn):
        f=os.path.basename(fn)
        
        if not f.endswith(self.ending):
            print('not f.endswith(self.ending)')
            exit()
        
        f=f[:-len(self.ending)]+'.xml'
        f=os.path.join(self.xmllabeldir,f) 
        
        return f


    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')

        label=self.labels[idx]

        if self.transform:
            image = self.transform(image)

        #print(image.size())

        sample = {'image': image, 'label': label, 'filename': self.imgfilenames[idx]}

        return sample


