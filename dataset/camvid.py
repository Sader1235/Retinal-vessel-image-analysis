import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from dataset.pre_processing import my_PreProc
import cv2
import random
# class_weight = torch.FloatTensor([0.25, 0.25, 0.25,1])
class_weight = torch.FloatTensor([0.0057471264, 0.0050251, 0.00884955752,1])

# mean = [0.611, 0.506, 0.54]
mean = [0.6127558736339982,0.5071148744673234,0.5406509545283443]

std = [0.13964046123851956,0.16156206296516235,0.165885041027991]

testmean = [0.6170943891910641,0.5133861905981716,0.545347489522038]
teststd = [0.14098655787705194,0.16313775003634445,0.16636559984060037]
class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
]


def _make_dataset(dir, Gray):
    names = []
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('RGB.png'):
                if Gray:
                    fname = fname.replace('_RGB', '')
                else:
                    fname = fname
                path = os.path.join(root, fname)
                name = path.split('/')[-1][:-8]#.split('_')[0]
                # print(path)
                # print(name) 
                names.append(name)
                images.append(path)
    return images, names

def _make_dataset_unlabelTR(dir, Gray, index_batch):
    names = []
    images = []
    fnames = [name for name in os.listdir(dir) if name.endswith('_RGB.png')]
    for fnames_num in range(len(fnames)):
        fname = fnames[fnames_num]
        if Gray:
            fname = fname.replace('_RGB.png', '.jpg')
        else:
            fname = fname
        path = os.path.join(dir, fname)
        name = path.split('/')[-1][:-8]
        # print(path, name)
        names.append(name)
        images.append(path)
    return images, names

def _make_dataset_unlabel_RITE(dir, Gray, index_batch):
    names = []
    images = []
    fnames = [name for name in os.listdir(dir) if name.endswith('_RGB.png')]
    for fnames_num in range(len(fnames)):
        fname = fnames[fnames_num]
        if Gray:
            fname = fname.replace('_RGB.png', '.jpg')
        else:
            fname = fname
        path = os.path.join(dir, fname)
        # name = path.split('/')[-1].split('_')[0]
        name = path.split('/')[-1][:-8]
        print(path, name)
        names.append(name)
        images.append(path)
    return images, names

def _make_dataset_unlabel(dir, Gray, index_batch):
    names = []
    images = []
    paths = os.listdir(dir)
    for path_num in range(len(paths)):
        img_path = os.path.join(dir, paths[path_num])
        fnames = [name for name in os.listdir(img_path) if name.endswith('_RGB.png')]
        for fnames_num in range(len(fnames)):
            fname = fnames[fnames_num]
            if Gray:
                fname = fname.replace('_RGB.png', '.jpg')
            else:
                fname = fname
            path = os.path.join(dir, paths[path_num], fname)
            name = path.split('/')[-1][:-8]
            # name = path.split('/')[-1].split('_')[0]
            print('------------', path, name)
            names.append(name)
            images.append(path)
    return images, names

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic)#.long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous()#.long()
        return label

class CamVid(data.Dataset):
    def __init__(self, root, Gray, joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(), loader=default_loader):
        self.root = root
        self.Gray = Gray
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        # print('self.root', self.root)
        self.imgs, self.names = _make_dataset(self.root, self.Gray)
    def __getitem__(self, index):
        path = self.imgs[index]
        name = self.names[index]
        if self.Gray:
            img = self.loader(path).convert('L')
            target = Image.open(os.path.join(self.root, name + '_manual1.png'))
        else:
            img = self.loader(path)
            target = Image.open(os.path.join(self.root, name + '.png')).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])
        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(target)/255
        # print('img',img.data.numpy().shape, np.max(img.data.numpy()))
        # print('target',name, target.shape, np.unique(target))
        return img, target, name
    def __len__(self):
        return len(self.imgs)
class CamVid_patch(data.Dataset):

    def __init__(self, root, Gray, joint_transform=None,
                 transform=None, target_transform=LabelToLongTensor(), loader=default_loader):
        self.root = root
        self.Gray = Gray
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.imgs, self.names = _make_dataset(self.root, self.Gray)
        self.width, self.height = 256, 256
    def __getitem__(self, index):
        path = self.imgs[index]
        name = self.names[index]
        if self.Gray:
            img = self.loader(path).convert('L')
            target = Image.open(os.path.join(self.root, name + '_manual1.png'))
            img_patches = Get_patches(img, patchH = 256, patchW = 256, cha = 'L')
        else:
            img = self.loader(path)
            target = Image.open(os.path.join(self.root, name + '_manual1.png'))
            img_patches = Get_patches(img, patchH = 256, patchW = 256, cha = 'RGB')
        
        target_patches = Get_patches(target, patchH = 256, patchW = 256, cha = 'L')
        # print('np.max(imgpatch)', np.max(np.array(img)))
        # print('np.max(targetpatch)', np.max(np.array(target)))
        return np.array(img_patches), np.array(target_patches), name

    def __len__(self):
        return len(self.imgs)
class Unlable_CamVid(data.Dataset):
    def __init__(self, root, Gray, index_start = 0, index_batch = 0, joint_transform=None,transform=None, target_transform=LabelToLongTensor(), loader=default_loader):
        self.root = root
        self.Gray = Gray
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        
        # print(self.root)
        if index_batch == 75:
            self.index_end = 300
            self.imgs_all, self.names_all = _make_dataset_unlabel(self.root, self.Gray, index_batch)
        if index_batch == 149:
            self.index_end = 149
            self.imgs_all, self.names_all = _make_dataset_unlabelTR(self.root, self.Gray, index_batch)
        if index_batch == 1475:
            self.index_end = 1475
            self.imgs_all, self.names_all = _make_dataset_unlabelTR(self.root, self.Gray, index_batch)
        if index_batch == 1624:
            self.index_end = 1624
            self.imgs_all, self.names_all = _make_dataset_unlabelTR(self.root, self.Gray, index_batch)
        if index_batch == -1:
            self.imgs_all, self.names_all = _make_dataset_unlabelTR(self.root, self.Gray, index_batch)
        if index_batch == -2:
            self.imgs_all, self.names_all = _make_dataset_unlabel_RITE(self.root, self.Gray, index_batch)
        # else:
            # self.index_end = index_start + 10*2**index_batch#40,80,160
            # self.imgs_all, self.names_all = _make_dataset_unlabel(self.root, self.Gray, index_batch)
        if index_batch > 0 :
            self.imgs = self.imgs_all[index_start : self.index_end]
            self.names = self.names_all[index_start : self.index_end]
        else :
            self.imgs = self.imgs_all
            self.names = self.names_all
        # print('len',len(self.imgs))
        # self.width, self.height = 256, 256
    def __getitem__(self, index):
        path = self.imgs[index]
        name = self.names[index]
        if self.Gray:
            img = self.loader(path).convert('L')
            # print(self.root, path, name + '.png')
            target = Image.open(os.path.join(self.root, path.replace('jpg', 'png')))
        else:
            img = self.loader(path)
            # target = Image.open(os.path.join(self.root, name + '.jpg'))
            target = Image.open(path).convert('L')
        # target = img

        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])
        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(target)/255
        # target = np.array(target)/255
        # print('img',name, img.data.numpy().shape, np.max(img.data.numpy()))
        # print('target',name, target.shape,np.unique(target))#C0018165 (3, 512, 512) 1.0
        return img, target, name

    def __len__(self):
        return len(self.imgs)
class Unlable_patch_CamVid(data.Dataset):
    def __init__(self, root, Gray, index_start = 0, index_batch = 0, joint_transform=None,transform=None, target_transform=LabelToLongTensor(), loader=default_loader):
        self.root = root
        self.Gray = Gray
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.imgs_all, self.names_all = _make_dataset_unlabel(self.root, self.Gray, index_batch)
        # print(os.path.join(self.root, self.split))
        if index_batch == 75:
            self.index_end = 302
        else:
            self.index_end = index_start + 10*2**index_batch#40,80,160
        self.imgs = self.imgs_all[index_start : self.index_end]
        self.names = self.names_all[index_start : self.index_end]
        print('len',index_start, self.index_end)
        self.width, self.height = 256, 256
    def __getitem__(self, index):
        path = self.imgs[index]
        name = self.names[index]
        if self.Gray:
            img = self.loader(path).convert('L')
            target = Image.open(os.path.join(self.root, path.split('/')[-2], name + '.png'))
        else:
            img = self.loader(path)
            target = Image.open(os.path.join(self.root, path.split('/')[-2], name + '.png'))
        img_patches = Get_patches(img, patchH = 256, patchW = 256)
        target_patches = Get_patches(target, patchH = 256, patchW = 256)
        print('np.max(imgpatch', np.max(np.array(img)))
        print('np.max(targetpatch)', np.max(np.array(target)))
        return np.array(img_patches), np.array(target_patches), name

    def __len__(self):
        return len(self.imgs)
def Get_patches(img, patchH, patchW, cha):
    patches = []#np.empty((patchNumber,patch224,patch224,imgChannel))
    # print(img.size)#512,512
    if cha == 'RGB':
        img_copy = Image.new('RGB', (img.size[0], img.size[1]))
    if cha == 'L':
        img_copy = Image.new('L', (img.size[0], img.size[1]))
    patch1 = img.crop((0, 0, patchH, patchW))
    # patch1.save('1.png')
    img_copy.paste(patch1, (0, 0, patchH, patchW))
	
    patch2 = img.crop((0, patchH, patchW, img.size[0]))
    # patch2.save('2.png')
    img_copy.paste(patch2, (0, patchH, patchW, img.size[0]))
	
    patch3 = img.crop((patchH, 0, img.size[0], patchW))
    # patch3.save('3.png')
    img_copy.paste(patch3, (patchH, 0, img.size[0], patchW))
	
    patch4 = img.crop((patchH, patchW, img.size[0], img.size[0]))
    # patch4.save('4.png')
    img_copy.paste(patch4, (patchH, patchW, img.size[0], img.size[0]))
    if cha == 'RGB':
        patch1 = np.transpose(np.array(patch1), (2,0,1))
        patch2 = np.transpose(np.array(patch2), (2,0,1))
        patch3 = np.transpose(np.array(patch3), (2,0,1))
        patch4 = np.transpose(np.array(patch4), (2,0,1))

        # print('patch shape:', cha , patch1.shape)
    
    patches.append(np.array(patch1))
    patches.append(np.array(patch2))
    patches.append(np.array(patch3))
    patches.append(np.array(patch4))
    return patches
