import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
import pickle
from render_sketch_chairv2 import redraw_Quick2RGB

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SLFIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode

        if hp.dataset_name == "ChairV2":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', 'ChairV2')
        elif hp.dataset_name == "ShoeV2":
            self.root_dir = os.path.join(hp.root_dir, 'Dataset', 'ShoeV2')

        with open(os.path.join(self.root_dir, hp.dataset_name + '_' + "Coordinate"), 'rb') as fp:
            self.Coordinate = pickle.load(fp)
       
        self.Skecth_Train_List = [x for x in self.Coordinate if 'train' in x]
        self.Skecth_Test_List = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')
        

    def __getitem__(self, item):
        sample = {}
        if self.mode == 'Train':
            sketch_path = self.Skecth_Train_List[item]

            positive_name = '_'.join(self.Skecth_Train_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_name + '.png')

            possible_list = list(range(len(self.Skecth_Train_List)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)]
            negative_name = '_'.join(self.Skecth_Train_List[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_name + '.png')
            
            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)          

            sketch_img = Image.fromarray(sketch_img[-1]).convert('RGB')
            positive_img = Image.open(positive_path)
            negative_img = Image.open(negative_path)                     

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)
            
            sample = {'sketch_img': sketch_img, 'positive_img': positive_img, 'negative_img': negative_img, 
                        'sketch_path': sketch_path, 'positive_path': positive_name, 'negative_path': negative_name
                        }
            
        elif self.mode == 'Test':

            sketch_path = self.Skecth_Test_List[item]

            positive_name = '_'.join(self.Skecth_Test_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_name + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)

            sketch_img = self.test_transform(Image.fromarray(sketch_img[-1]).convert('RGB'))
            positive_img = self.test_transform(Image.open(positive_path))

            sample = {'sketch_img': sketch_img, 'positive_img': positive_img, 
                        'sketch_path': sketch_path, 'positive_path': positive_name}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Skecth_Train_List)
        elif self.mode == 'Test':
            return len(self.Skecth_Test_List)


def get_dataloader(hp):
    dataset_Train = SLFIR_Dataset(hp, mode='Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True, num_workers=int(hp.nThreads))

    dataset_Test = SLFIR_Dataset(hp, mode='Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False, num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test


def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(320), transforms.RandomCrop(299)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)

