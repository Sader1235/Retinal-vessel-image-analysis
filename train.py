from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import models.unet_model as models
from multiprocessing import Pool 
# from models.loss import CrossEntropy2d
# import dataset.cifar10 as dataset
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
# from tensorboardX import SummaryWriter
from pathlib import Path
from dataset import camvid, joint_transforms
import utils.imgs

parser = argparse.ArgumentParser(description='PyTorch AV Training')
# Optimization options
parser.add_argument('--epochs', default=2000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--val-iteration', type=int, default=10, help='Number of labeled data')
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Use CUDA
use_cuda = torch.cuda.is_available()
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
###################
# va = 'un'
# va= 'un'
va= 'train'
PATH = 'RGB/512/'
start_epoch = 0
inch = 3
class_num = 4
LoadData = True
Gray_Flag = False
batch_size = 24
###################
best_dice = 0.  # best test accuracy
best_acc = 0.  # best test accuracy
def main():
    ##################################################################data
    transform_train = transforms.Compose([transforms.ToTensor()])

    train_joint_transformer = transforms.Compose([joint_transforms.JointRandomHorizontalFlip()])
	#train
    train_dset = camvid.Unlable_CamVid(unlabel_PATH, Gray = Gray_Flag, index_start = 0, index_end = 120, joint_transform=train_joint_transformer, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
	#val
    train_val_dset = camvid.Unlable_CamVid(unlabel_PATH, Gray = Gray_Flag, index_start = 120, index_end = 149, joint_transform=train_joint_transformer, transform=transform_train)
    train_val_loader = torch.utils.data.DataLoader(train_val_dset, batch_size=batch_size, shuffle=True)
	#test
    train_test_dset = camvid.Unlable_test_CamVid(unlabel_test_PATH, Gray = Gray_Flag, index_start = 0, index_end = 1475, joint_transform=train_joint_transformer, transform=transform_train)
    train_test_loader = torch.utils.data.DataLoader(train_test_dset, batch_size=batch_size, shuffle=True)


    visuleize = True
    if visuleize:
        print("Train: %d" %len(train_loader.dataset.imgs))
        # print("Test: %d" %len(train_loader.dataset.imgs))
        inputs, targets, names = next(iter(train_loader))
        # inputs, targets, names = next(iter(un_loader))
        print("Inputs: ", inputs.size())
        print("Targets: ", targets.size())
        
        utils.imgs.view_image_Gray(inputs[0])
        utils.imgs.view_annotated(targets[0])  
    makedir_Flag = True
    if makedir_Flag:
        dirs = ['./result/' + PATH + 'visulizeT0/','./result/' + PATH + 'visulizeT1/','./result/' + PATH + 'visulizeT2/']
        for dir_num in range(len(dirs)):
            os.makedirs(dirs[dir_num], exist_ok = True)
    ########################
    global best_dice
    global best_acc

    device = torch.device("cuda")
    def create_model(ema=False):
        model = models.UNet4(n_channels=inch, n_classes=class_num)
        # model.apply(init)
        model = torch.nn.DataParallel(model).to(device).cuda()
        return model
    model = create_model()
    model = torch.load('')
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    test_dices = []
    test_aucs = []
    model_fold = "./PKL/" + PATH
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('--------------------------------------')
        ###########################
        
        #####################
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        if va == 'un':
            print('val', val)
            train_loss, train_dice_x1, train_dice_x2, train_dice_x3 = train(un_loader, model, optimizer, epoch, dirs[0])
            test_loss, test_dice1, test_dice2, test_dice3 = validate(un_loader, model, epoch, dirs[1])
            test_loss, test_dice1, test_dice2, test_dice3 = validate(un_loader_val, model, epoch, dirs[1])
        else:
            print('val', va)
            train_loss, train_dice_x1, train_dice_x2, train_dice_x3, train_acc = train(train_loader, model, optimizer, epoch, dirs[0])
            val_loss, val_dice1, val_dice2, val_dice3, val_acc = validate(train_val_loader, model, epoch, dirs[1])
            test(train_test_loader, model, epoch, dirs[2])
        
        # save model
        monitor_dice = val_dice1 + val_dice2 + val_dice3
        monitor_acc = val_acc

        is_best_dice = monitor_dice > best_dice
        is_best_acc = monitor_acc > best_acc

        best_dice = max(monitor_dice, best_dice)
        best_acc = max(monitor_acc, best_acc)

        if is_best_dice or is_best_acc:
            name = model_fold + str(epoch) + '_D1-' + str(round(val_dice1,4)) + '_D2-' + str(round(val_dice2,4))+ '_D3-' + str(round(val_dice3,4))+ '_A-' + str(round(val_acc,4)) + '.pkl'
            print(name)
            torch.save(model, name)
    print('Best dice:', best_dice)
    print('Best acc:', best_acc)

def Regular_Threshod(epoch, Guess_label, inputs_u, targets_u0, name_u, Mask_Rate = 0.1, mode = 'val'):
    N = int((1- Mask_Rate) * Guess_label.shape[1] * Guess_label.shape[2])
    # print(N, Guess_label.shape)
    Guess_label_view_list = 0
    for batch_num in range(Guess_label.shape[0]):
        Guess_label_view = sorted(Guess_label[batch_num].data.cpu().numpy().flatten())#0-1
        Guess_label_view_list = Guess_label_view_list + Guess_label_view[N]
        # print('Guess_label', N, Guess_label_view[N])
    Threshod = np.round(Guess_label_view_list / Guess_label.shape[0], 4)

    
    return Threshod
def train(trainloader, model, optimizer, epoch, dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    dices_x1 = AverageMeter()
    dices_x2 = AverageMeter()
    dices_x3 = AverageMeter()
    accs = AverageMeter()
    Threshod = 0.7
    train_iter = iter(trainloader)
    data_list = []
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, name_x = train_iter.next()
        except:
            train_iter = iter(trainloader)
            inputs_x, targets_x, name_x = train_iter.next()
        batch_size = inputs_x.size(0)

        inputs_x, targets_x = torch.FloatTensor(inputs_x.float()).cuda(), torch.FloatTensor(targets_x.float()).cuda().long()
        outputs = model(inputs_x)
        loss = loss_calc(outputs, targets_x)
        # print('shape', inputs_x.shape, outputs.shape, targets_x.shape)#torch.Size([4, 3, 512, 512]) torch.Size([4, 4, 512, 512]) torch.Size([4, 512, 512])
        pred = get_predictions(outputs)
        x_dice1, x_dice2, x_dice3 = get_dice(pred, targets_x, Threshod)
        # print('dice', x_dice1, x_dice2, x_dice3)
        ###############
        # data_list.append([targets_x.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten()])
        # pool = Pool() 
        
        # m_list = pool.map(f, data_list)
        # pool.close() 
        # pool.join() 
        # for m in m_list:
            # ConfM.addM(m)
        ###############
        acc = 1 - error(pred, targets_x.data.cpu())
        # record loss
        dices_x1.update(x_dice1, inputs_x.size(0))
        dices_x2.update(x_dice2, inputs_x.size(0))
        dices_x3.update(x_dice3, inputs_x.size(0))
        losses.update(loss.item(), inputs_x.size(0))
        accs.update(acc, inputs_x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #save output
        if random.random() > 0.5:
            visulize(name_x, inputs_x, pred, targets_x, epoch, dir, Threshod)
    print('train---Loss: {loss:.4f} | Dice_x1: {dice_x1:.4f}| Dice_x2: {dice_x2:.4f} | Dice_x3: {dice_x3:.4f} | acc: {acc:.4f}'.format(loss=losses.avg, dice_x1=dices_x1.avg, dice_x2=dices_x2.avg, dice_x3=dices_x3.avg, acc=accs.avg))
    return (losses.avg, dices_x1.avg, dices_x2.avg, dices_x3.avg, accs.avg)

def validate(valloader, model, epoch, dir):
    losses = AverageMeter()
    dices1 = AverageMeter()
    dices2 = AverageMeter()
    dices3 = AverageMeter()
    accs = AverageMeter()
    new_pred = []
    new_target = []
    data_list = []
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    model.eval()
    Threshod = 0.2
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets, names) in enumerate(valloader):
            # measure data loading time
            inputs, targets = torch.FloatTensor(inputs.float()).cuda(), torch.FloatTensor(targets.float()).cuda().long()
            outputs = model(inputs)
            loss = loss_calc(outputs, targets)
            # print('shape', inputs.shape, outputs.shape, targets.shape)#torch.Size([4, 3, 512, 512]) torch.Size([4, 4, 512, 512]) torch.Size([4, 512, 512])
            pred = get_predictions(outputs)
            x_dice1, x_dice2, x_dice3 = get_dice(pred, targets, Threshod)
            # print('dice', x_dice1, x_dice2, x_dice3)
            ###############
            # data_list.append([targets.data.cpu().numpy().flatten(), pred.data.cpu().numpy().flatten()])
            # pool = Pool() 
            
            # m_list = pool.map(f, data_list)
            # pool.close() 
            # pool.join() 
            # for m in m_list:
                # ConfM.addM(m)
            # acc = ConfM.accuracy()
            ###############
            acc = 1 - error(pred, targets.data.cpu())
            # record loss
            dices1.update(x_dice1, inputs.size(0))
            dices2.update(x_dice2, inputs.size(0))
            dices3.update(x_dice3, inputs.size(0))
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))
            #save output
            if random.random() > 0.5:
                visulize(names, inputs, pred, targets, epoch, dir, Threshod)
        print('val---Loss: {loss:.4f} | Dice1: {dice1:.4f}| Dice2: {dice2:.4f} | Dice3: {dice3:.4f}| acc: {acc:.4f}'.format(loss=losses.avg, dice1=dices1.avg, dice2=dices2.avg, dice3=dices3.avg, acc=accs.avg))
    return (losses.avg, dices1.avg, dices2.avg, dices3.avg, accs.avg)
def test(valloader, model, epoch, dir):
    losses = AverageMeter()
    dices1 = AverageMeter()
    dices2 = AverageMeter()
    dices3 = AverageMeter()
    accs = AverageMeter()
    new_pred = []
    new_target = []
    data_list = []
    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    model.eval()
    Threshod = 0.2
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, names) in enumerate(valloader):
            # measure data loading time
            inputs = torch.FloatTensor(inputs.float()).cuda()
            outputs = model(inputs)
            
            # print('shape', inputs.shape, outputs.shape, targets.shape)#torch.Size([4, 3, 512, 512]) torch.Size([4, 4, 512, 512]) torch.Size([4, 512, 512])
            pred = get_predictions(outputs)
            
            if random.random() > 0.5:
                visulize_test(names, inputs, pred, epoch, dir, Threshod)

def visulize(a, b, c, d, epoch, dir, Threshod):
    
    #a, b, c, d= names, inputs, outputs, targets
    if epoch % 1 == 0:
        for batch_num in range(len(a)):
            batch_num = 0
            name_save = a[batch_num]
            inputs_save = b[batch_num].data.squeeze(0).cpu().numpy()
            if inputs_save.shape[0] == 3:
                inputs_save = np.transpose(inputs_save, (1,2,0))[:,:,::-1]
            outputs_save = c[batch_num].data.squeeze(0).cpu().numpy()
            targets_save = d[batch_num].data.squeeze(0).cpu().numpy()
            # print('shape0:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save))#(3, 512, 512) (512, 512) (512, 512) 0.9725647 1.0
            outputs_save = colorize_mask(outputs_save, class_num).astype(np.int32)
            targets_save = colorize_mask(targets_save, class_num).astype(np.int32)
            # print('shape1:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save), np.max(targets_save))
            if outputs_save.shape[0] == 3:
                outputs_save = np.transpose(outputs_save, (1,2,0))#[:,:,::-1]
                targets_save = np.transpose(targets_save, (1,2,0))#[:,:,::-1]
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Inp.png', 255 * inputs_save)
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Out.png', outputs_save)
            
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Tar.png', targets_save)
def visulize_test(a, b, c, epoch, dir, Threshod):
    
    #a, b, c, d= names, inputs, outputs, targets
    if epoch % 1 == 0:
        for batch_num in range(len(a)):
            batch_num = 0
            name_save = a[batch_num]
            inputs_save = b[batch_num].data.squeeze(0).cpu().numpy()
            if inputs_save.shape[0] == 3:
                inputs_save = np.transpose(inputs_save, (1,2,0))[:,:,::-1]
            outputs_save = c[batch_num].data.squeeze(0).cpu().numpy()

            outputs_save = colorize_mask(outputs_save, class_num).astype(np.int32)
            # print('shape1:', inputs_save.shape, outputs_save.shape, targets_save.shape, np.max(inputs_save), np.max(outputs_save), np.max(targets_save))
            if outputs_save.shape[0] == 3:
                outputs_save = np.transpose(outputs_save, (1,2,0))
                
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Inp.png', 255 * inputs_save)
            cv2.imwrite(dir + str(epoch) + '_' + str(name_save) + '_Out.png', outputs_save)

def For_dice(y_true, y_pred, N, index):
    pred_copy = torch.zeros((N, y_pred.shape[1],y_pred.shape[2]))
    true_copy = torch.zeros((N, y_true.shape[1],y_true.shape[2]))
    pred_copy[y_pred == index] = 1
    true_copy[y_true == index] = 1
    true_copy, pred_copy = true_copy.data.cpu().numpy(), pred_copy.data.cpu().numpy()
    # print('max1',np.max(true_copy), np.max(pred_copy))#1 1
    dice = 2 * (pred_copy * true_copy).sum(1).sum(1) / (pred_copy.sum(1).sum(1) + true_copy.sum(1).sum(1) + 1e-5)
    dice = dice.sum() / N
    # print(dice)
    return dice
def get_dice(y_true, y_pred, Threshod):
    smooth = 1e-5
    dices = []
    N = y_true.shape[0]
    # print('max0',N, np.max(y_true.data.cpu().numpy()), np.max(y_pred.data.cpu().numpy()))#3 3
    for index in range(1,4):#1,2,3 classnum
        
        dice = For_dice(y_true, y_pred, N, index)
        dices.append(dice)
    
    return dices[0], dices[1], dices[2]

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    return criterion(pred, label)

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in xrange(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in xrange(self.nclass):
            jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        # print(gt.shape, pred.shape)
        # print(len(gt), len(pred))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m

def colorize_mask(mask, class_num):#R=3,G=2,B=1
    # mask: numpy array of the mask
    
    # print('mask.shape', mask.shape)
    mask_save = np.zeros((class_num -1, mask.shape[0], mask.shape[1]))
    # print('mask_save.shape', mask_save.shape)
    for index in range(1, class_num):
        mask_copy = np.zeros((mask.shape[0], mask.shape[1]))
        # print('mask_copy.shape', mask_copy.shape)
        mask_copy[mask == index] = 255
        mask_save[index-1,:,:] = mask_copy

    return mask_save
def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    # n_pixels = 1
    incorrect = preds.ne(targets).cpu().sum().numpy()
    err = incorrect/n_pixels
    # print(incorrect,n_pixels,err)
    # return round(err,5)
    return err
	
def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data, 0.15)
        nn.init.constant_(module.bias.data, 0)
if __name__ == '__main__':
    main()
