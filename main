# encoding: utf-8


import argparse
import os
import shutil
import socket
import time
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pytorch_wavelets import DWTForward
import transformed as transforms
from ImageFolderDataset import MyImageFolder
from hidernet import HidNet
from RecoveryNet import Recoverynet1,Recoverynet2,Recoverynet3

import numpy as np
from scipy.stats import ortho_group
import  PIL.Image as Image


DATA_DIR = 'E:/Xishun/imagehide/data'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=20,
                    help='input batch size')
parser.add_argument('--imageSize1', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--imageSize', type=int, default=128,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=1000,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.9 each decay_round')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--resume', default=True,
                    help="path to resume model")
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet1', default='',
                    help="path to Recovernet1 (to continue training)")
parser.add_argument('--Rnet2', default='',
                    help="path to Recovernet2 (to continue training)")
parser.add_argument('--Rnet3', default='',
                    help="path to Recovernet3 (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=1.01,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=30, help='the frequency of save the resultPic')



def change(x,k,l):
    x = np.squeeze(x)

    x_j = x.swapaxes(0, 2)
    x_j = x_j.swapaxes(0, 1)
    x_j = x_j * 255
    x_j = Image.fromarray(np.uint8(x_j))
    x_j = x_j.resize((k, l))
    x_j = np.array(x_j)
    x_j = x_j.swapaxes(1, 2)
    x_j = x_j.swapaxes(0, 1)
    x_j = x_j / 255
    x_j = torch.Tensor(x_j)
    x = x_j.unsqueeze(0)
    return x

class myresize1():
    def Myresize(x,k,l):
        x1 = x[0:1,::]
        x1 = change(x1,k,l)

        return x1


class myresize():
    def Myresize(x,k,l):
        x1 = x[0:1,::]
        x1 = change(x1,k,l)

        return x1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)

    cur_work_dir, mainfile = os.path.split(main_file_path)
    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)
    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)





def main():
    ############### define global parameters ###############batch_size
    global opt, optimizerH, optimizerR1,optimizerR2,optimizerR3, writer, batch_size, logPath,  \
        schedulerH, schedulerR1,schedulerR2,schedulerR3,train_dataset1, val_dataset1,  val_loader, smallestLoss, val_dataset, \
        test_loader1,test_loader,train_dataset,test_dataset1, test_dataset, val_sumloss, val_hloss, val_rloss, epoch, start_epoch

    #################  output configuration   ###############
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True


    ############  create dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if opt.test == '':
        writer = SummaryWriter(comment='**' + opt.remark)
        writer = SummaryWriter(comment='_' + opt.remark)
        ##############   get dataset   ############################
        traindir1 = os.path.join(DATA_DIR, 'train1')
        traindir = os.path.join(DATA_DIR, 'train')
        valdir1 = os.path.join(DATA_DIR, 'val1')
        valdir = os.path.join(DATA_DIR, 'val')
        train_dataset1 = MyImageFolder(
            traindir1,
            transforms.Compose([
                transforms.Resize([opt.imageSize1, opt.imageSize1]),
                transforms.ToTensor(),
            ]))
        assert train_dataset1
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert train_dataset
        val_dataset1 = MyImageFolder(
            valdir1,
            transforms.Compose([
                transforms.Resize([opt.imageSize1, opt.imageSize1]),
                transforms.ToTensor(),
            ]))
        assert val_dataset1
        val_dataset = MyImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))

        assert val_dataset
    else:
        opt.Hnet = ""
        opt.Rnet = ""
        testdir1 = os.path.join(DATA_DIR, 'test1')
        testdir = os.path.join(DATA_DIR, 'test')
        test_dataset1 = MyImageFolder(
            testdir1,
            transforms.Compose([
                transforms.Resize([opt.imageSize1, opt.imageSize1]),
                transforms.ToTensor(),
            ]))
        assert test_dataset
        test_dataset = MyImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert test_dataset

    Hnet = HidNet(input_nc=10, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)

    if opt.Yunet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)

    Rnet1 = Recoverynet1(batch_size, output_function=nn.Tanh)
    Rnet1.cuda()
    Rnet1.apply(weights_init)
    if opt.Rnet != "":
        Rnet1.load_state_dict(torch.load(opt.Rnet1))
    if opt.ngpu > 1:
        Rnet1 = torch.nn.DataParallel(Rnet1).cuda()
    print_network(Rnet1)

    Rnet2 = Recoverynet2(batch_size, output_function=nn.Tanh)
    Rnet2.cuda()
    Rnet2.apply(weights_init)
    if opt.Rnet2 != "":
        Rnet2.load_state_dict(torch.load(opt.Rnet2))
    if opt.ngpu > 1:
        Rnet2= torch.nn.DataParallel(Rnet2).cuda()
    print_network(Rnet2)

    Rnet3 = Recoverynet3(batch_size, output_function=nn.Tanh)
    Rnet3.cuda()
    Rnet3.apply(weights_init)
    if opt.Rnet != "":
        Rnet3.load_state_dict(torch.load(opt.Rnet3))
    if opt.ngpu > 1:
        Rnet3 = torch.nn.DataParallel(Rnet3).cuda()
    print_network(Rnet3)

    batch_size = opt.batchSize

    criterion = nn.MSELoss().cuda()

    if opt.test == '':
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        optimizerR1 = optim.Adam(Rnet1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR1 = ReduceLROnPlateau(optimizerR1, mode='min', factor=0.9, patience=8, verbose=True)

        optimizerR2 = optim.Adam(Rnet2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR2 = ReduceLROnPlateau(optimizerR2, mode='min', factor=0.9, patience=8, verbose=True)

        optimizerR3 = optim.Adam(Rnet3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR3 = ReduceLROnPlateau(optimizerR3, mode='min', factor=0.9, patience=8, verbose=True)

        train_loader1 = DataLoader(train_dataset1, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers),drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers),drop_last=True)
        val_loader1 = DataLoader(val_dataset1, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers),drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers),drop_last=True)
        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)


        if os.path.isfile(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/Hcheckpoint_model_epoch_95.pth'):
            checkpoint = torch.load(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/Hcheckpoint_model_epoch_95.pth')
            Hnet.load_state_dict(checkpoint['model'])
            optimizerH.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        if os.path.isfile(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R1checkpoint_model_epoch_95.pth'):
            checkpoint = torch.load(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R1checkpoint_model_epoch_95.pth')
            Rnet1.load_state_dict(checkpoint['model'])
            optimizerR1.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')

        if os.path.isfile(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R2checkpoint_model_epoch_95.pth'):
            checkpoint = torch.load(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R2checkpoint_model_epoch_95.pth')
            Rnet2.load_state_dict(checkpoint['model'])
            optimizerR2.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        if os.path.isfile(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R3checkpoint_model_epoch_95.pth'):
            checkpoint = torch.load(
                'E:/Xishun/imagehide/training/DESKTOP-M9QJC45_2021-01-23-13_07_54/checkPoints/R3checkpoint_model_epoch_95.pth')
            Rnet3.load_state_dict(checkpoint['model'])
            optimizerR3.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
            # -----------------------加载开始进行训练--------------------------------

        for epoch in range(start_epoch, opt.niter):
        # for epoch in range(opt.niter):
            errH,errR1,errR2,errR3=train(train_loader1,train_loader, epoch, Hnet=Hnet, Rnet1=Rnet1, Rnet2=Rnet2, Rnet3=Rnet3, criterion=criterion)

            # -----------validation验证集开始进行--------------------------
            validation(val_loader1,val_loader, epoch, Hnet=Hnet, Rnet1=Rnet1, Rnet2=Rnet2, Rnet3=Rnet3,
                                                           criterion=criterion)
            # --------------adjust learning rate调整学习率----------------
            schedulerH.step(errH)
            schedulerR1.step(errR1)
            schedulerR2.step(errR2)
            schedulerR3.step(errR3)
            # --------------加载保存网络的路径和名称--------------------------
            # 每5次进行一次保存


            if epoch % 5 == 0 and epoch != 0:
                Hfilepath = os.path.join(opt.outckpts, 'Hcheckpoint_model_epoch_{}.pth'.format(epoch))
                Hfilepath = Hfilepath.replace('\\', '/')
                Hfilepath = 'E:/Xishun/imagehide' + Hfilepath

                R1filepath = os.path.join(opt.outckpts, 'R1checkpoint_model_epoch_{}.pth'.format(epoch))
                R1filepath = R1filepath.replace('\\', '/')
                R1filepath = 'E:/Xishun/imagehide' + R1filepath

                R2filepath = os.path.join(opt.outckpts, 'R2checkpoint_model_epoch_{}.pth'.format(epoch))
                R2filepath = R2filepath.replace('\\', '/')
                R2filepath = 'E:/Xishun/imagehide' + R2filepath

                R3filepath = os.path.join(opt.outckpts, 'R3checkpoint_model_epoch_{}.pth'.format(epoch))
                R3filepath = R3filepath.replace('\\', '/')
                R3filepath = 'E:/Xishun/imagehide' + R3filepath
                # -----------------------保存每次epoch的网络模型，参数----------------------

                Hstate = {'model': Hnet.state_dict(), 'optimizer': optimizerH.state_dict(), 'epoch': epoch}
                torch.save(Hstate, Hfilepath)
                print(Hfilepath)
                R1state = {'model': Rnet1.state_dict(), 'optimizer': optimizerR1.state_dict(), 'epoch': epoch}
                torch.save(R1state, R1filepath)
                print(R1filepath)
                R2state = {'model': Rnet2.state_dict(), 'optimizer': optimizerR2.state_dict(), 'epoch': epoch}
                torch.save(R2state, R2filepath)
                print(R2filepath)
                R3state = {'model': Rnet3.state_dict(), 'optimizer': optimizerR3.state_dict(), 'epoch': epoch}
                torch.save(R3state, R3filepath)
                print(R3filepath)

                if val_sumloss < globals()["smallestLoss"]:
                    globals()["smallestLoss"] = val_sumloss
                    # do checkPointing
                    torch.save(Hnet.state_dict(),
                               '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                                   opt.outckpts, epoch, val_sumloss, val_hloss))
                    torch.save(Rnet1.state_dict(),
                               '%s/netR_epoch_%d,sumloss=%.6f,R1loss=%.6f.pth' % (
                                   opt.outckpts, epoch, val_sumloss, val_rloss1))
                    torch.save(Rnet2.state_dict(),
                               '%s/netR_epoch_%d,sumloss=%.6f,R2loss=%.6f.pth' % (
                                   opt.outckpts, epoch, val_sumloss, val_rloss2))
                    torch.save(Rnet2.state_dict(),
                               '%s/netR_epoch_%d,sumloss=%.6f,R3loss=%.6f.pth' % (
                                   opt.outckpts, epoch, val_sumloss, val_rloss3))

        writer.close()

    # test mode
    else:
        test_loader1 = DataLoader(test_dataset1, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers),drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers),drop_last=True)
        test(test_loader1,test_loader, 0, Hnet=Hnet, Rnet1=Rnet1, Rnet2=Rnet2,Rnet3=Rnet3,criterion=criterion)
        print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def train(train_loader1, train_loader, epoch, Hnet, Rnet1,Rnet2,Rnet3, criterion):
    global cover_img, errH, errR1, errR2, errR3
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    R1losses = AverageMeter()
    R2losses = AverageMeter()
    R3losses = AverageMeter()
    SumLosses = AverageMeter()

    Hnet.train()
    Rnet1.train()
    Rnet2.train()
    Rnet3.train()

    start_time = time.time()
    for j, data in enumerate(train_loader1, 0):
        for i, data1 in enumerate(train_loader, 0):
            data_time.update(time.time() - start_time)
            Hnet.zero_grad()
            Rnet1.zero_grad()
            Rnet2.zero_grad()
            Rnet3.zero_grad()

            secret_img = data
            cover_img = data1
            print(secret_img.shape)
            secret_copy = copy.copy(secret_img)
            MDwt = DWTForward(J=1, mode='zero', wave='haar', )
            Alanl, Alanh = MDwt(secret_copy)
            A1 = Alanh[0][:, 0:1, ::]
            A2 = Alanh[0][:, 1:2, ::]
            A3 = Alanh[0][:, -1:, ::]
            A_1 = torch.squeeze(A1, dim=1)
            A_2 = torch.squeeze(A2, dim=1)
            A_3 = torch.squeeze(A3, dim=1)
            A = torch.cat([A_1, A_2], dim=1)
            A = torch.cat([A, A_3], dim=1)
            A = torch.cat([Alanl, A], dim=1)

            batch_size = opt.batchSize

            secret_copy = np.array(secret_copy)
            secret_64 = myresize.Myresize(secret_copy, 64,64)
            secret_128 = myresize.Myresize(secret_copy, 128,128)

            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                secret_64 = secret_64.cuda()
                secret_128 = secret_128.cuda()
                A = A.cuda()

            A = Variable(A)
            secret_imgv = Variable(secret_img)
            secret_64v = Variable(secret_64)
            secret_128v = Variable(secret_128)
            cover_imgv = Variable(cover_img)

            container_img = Hnet(A,cover_imgv)
            errH = criterion(container_img, cover_imgv)
            Hlosses.update(errH.item(),batch_size)

            container_img = container_img.detach()

            rev_secret_64,future1 = Rnet1(container_img)
            errR1 = criterion(rev_secret_64, secret_64v)
            R1losses.update(errR1.item(),batch_size)

            rev_secret_64 =  rev_secret_64.detach()
            future1 = future1.detach()

            rev_secret_128, future2 = Rnet2(rev_secret_64,future1)
            errR2 = criterion(rev_secret_128, secret_128v)
            R2losses.update(errR2.item(), batch_size)

            rev_secret_128 = rev_secret_128.detach()
            future2 = future2.detach()

            rev_secret = Rnet3(rev_secret_128,future2)
            errR3 = criterion(rev_secret, secret_imgv)
            R3losses.update(errR3.item(), batch_size)


            err_sum = errH + (errR1+errR2+errR3)/3
            SumLosses.update(err_sum.item(),batch_size)

            errH.backward()
            errR1.backward()
            errR2.backward()
            errR3.backward()
            err_sum.backward()

            optimizerH.step()
            optimizerR1.step()
            optimizerR2.step()
            optimizerR3.step()

            batch_time.update(time.time() - start_time)
            start_time = time.time()
            #
            log = '[%d/%d][%d/%d][%d/%d]\tLoss_H: %.4f Loss_R1: %.4f Loss_R2: %.4f Loss_R3: %.4fLoss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f'% \
                  ((epoch, opt.niter, j, len(train_loader1), i, len(train_loader),
                Hlosses.val, R1losses.val,R2losses.val,R3losses.val, SumLosses.val, data_time.val, batch_time.val))

            if i % opt.logFrequency == 0 and i!=0:
                print_log(log, logPath)
            else:
                print_log(log, logPath, console=False)

            # genereate a picture every resultPicFrequency steps
            if epoch % 5 == 0 and j % opt.resultPicFrequency == 0:
                save_result_pic(batch_size, cover_img, container_img, secret_img, rev_secret, epoch, j,
                                opt.trainpics)

            # else:
            #     continue



    # epcoh log
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR1_lr = %.8f  optimizerR2_lr = %.8f optimizerR3_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR1.param_groups[0]['lr'],optimizerR2.param_groups[0]['lr'],optimizerR3.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_R1loss=%.6f\t  epoch_R2loss=%.6f\t epoch_R3loss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, R1losses.avg, R2losses.avg,R3losses.avg,SumLosses.avg)
    print_log(epoch_log, logPath)


    if not opt.debug:
        # record lr
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R1_lr", optimizerR1.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R2_lr", optimizerR2.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R3_lr", optimizerR3.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # record loss
        writer.add_scalar('train/R1_loss', R1losses.avg, epoch)
        writer.add_scalar('train/R1_loss', R2losses.avg, epoch)
        writer.add_scalar('train/R1_loss', R3losses.avg, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)
    return errH,errR1,errR2,errR3

def validation(val_loader1, val_loader, epoch,  Hnet, Rnet1,Rnet2,Rnet3, criterion):
    global cover_img, batch_size, container_img, secret_img, rev_secret
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet1.eval()
    Rnet2.eval()
    Rnet3.eval()
    valHlosses = AverageMeter()
    valR1losses = AverageMeter()
    valR2losses = AverageMeter()
    valR3losses = AverageMeter()
    valSumLosses =AverageMeter()
    for j, data in enumerate(val_loader1, 0):
        for i, data1 in enumerate(val_loader, 0):

            Hnet.zero_grad()
            Rnet1.zero_grad()
            Rnet2.zero_grad()
            Rnet3.zero_grad()
            batch_size = opt.batchSize
            secret_img = data
            cover_img = data1
            secret_copy = copy.copy(secret_img)
            MDwt = DWTForward(J=1, mode='zero', wave='haar', )
            Alanl, Alanh = MDwt(secret_copy)
            A1 = Alanh[0][:, 0:1, ::]
            A2 = Alanh[0][:, 1:2, ::]
            A3 = Alanh[0][:, -1:, ::]
            A_1 = torch.squeeze(A1, dim=1)
            A_2 = torch.squeeze(A2, dim=1)
            A_3 = torch.squeeze(A3, dim=1)
            A = torch.cat([A_1, A_2], dim=1)
            A = torch.cat([A, A_3], dim=1)
            A = torch.cat([Alanl, A], dim=1)
            secret_copy = np.array(secret_copy)
            secret_64 = myresize.Myresize(secret_copy, 64, 64)
            secret_128 = myresize.Myresize(secret_copy, 128, 128)

            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                secret_64 = secret_64.cuda()
                secret_128 = secret_128.cuda()
                A = A.cuda()

            A = Variable(A)
            secret_imgv = Variable(secret_img)
            secret_64v = Variable(secret_64)
            secret_128v = Variable(secret_128)
            cover_imgv = Variable(cover_img)

            container_img = Hnet(A, cover_imgv)
            errH = criterion(container_img, cover_imgv)
            valHlosses.update(errH.item(), batch_size)

            container_img = container_img.detach()

            rev_secret_64, future1 = Rnet1(container_img)
            errR1 = criterion(rev_secret_64, secret_64v)
            valR1losses.update(errR1.item(), batch_size)

            rev_secret_64 = rev_secret_64.detach()
            future1 = future1.detach()

            rev_secret_128, future2 = Rnet2(rev_secret_64,future1)
            errR2 = criterion(rev_secret_128, secret_128v)
            valR2losses.update(errR2.item(), batch_size)

            rev_secret_128 = rev_secret_128.detach()
            future2 = future2.detach()

            rev_secret = Rnet3(rev_secret_128, future2)
            errR3 = criterion(rev_secret, secret_imgv)
            valR3losses.update(errR3.item(), batch_size)

            err_sum = errH + (errR1 + errR2 + errR3) / 3
            valSumLosses.update(err_sum.item(), batch_size)

        if j % 50 == 0:
                save_result_pic(batch_size, cover_img, container_img, secret_img, rev_secret, epoch, j,
                                opt.validationpics)



        val_time = time.time() - start_time
        val_log = "validation[%d] val_Hloss = %.6f\t val_R1loss = %.6f\t val_R2loss = %.6f\t val_R3loss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
            epoch, valHlosses.avg, valR1losses.avg,valR2losses.avg, valR3losses.avg,  valSumLosses.avg, val_time)
        print_log(val_log, logPath)

        if not opt.debug:
            writer.add_scalar('validation/H_loss_avg', valHlosses.avg, epoch)
            writer.add_scalar('validation/R1_loss_avg', valR1losses.avg, epoch)
            writer.add_scalar('validation/R2_loss_avg', valR2losses.avg, epoch)
            writer.add_scalar('validation/R3_loss_avg', valR3losses.avg, epoch)
            writer.add_scalar('validation/sum_loss_avg',  valSumLosses, epoch)

        print(
            "#################################################### validation end ########################################################")
        return  valHlosses, valR1losses,valR2losses, valR3losses,  valSumLosses

def test(test_loader1, test_loader, epoch, Hnet, Rnet1, Rnet2,Rnet3,criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet1.eval()
    Rnet2.eval()
    Rnet3.eval()
    testHlosses = AverageMeter()
    testR1losses = AverageMeter()
    testR2losses = AverageMeter()
    testR3losses = AverageMeter()
    testSumLosses =  AverageMeter()
    for i, data in enumerate(test_loader1, 0):
        for j, data1 in enumerate(test_loader, 0):

            Hnet.zero_grad()
            Rnet1.zero_grad()
            Rnet2.zero_grad()
            Rnet3.zero_grad()

            batch_size = opt.batchSize
            secret_img = data
            cover_img = data1
            secret_copy = copy.copy(secret_img)
            MDwt = DWTForward(J=1, mode='zero', wave='haar', )
            Alanl, Alanh = MDwt(secret_copy)
            A1 = Alanh[0][:, 0:1, ::]
            A2 = Alanh[0][:, 1:2, ::]
            A3 = Alanh[0][:, -1:, ::]
            A_1 = torch.squeeze(A1, dim=1)
            A_2 = torch.squeeze(A2, dim=1)
            A_3 = torch.squeeze(A3, dim=1)
            A = torch.cat([A_1, A_2], dim=1)
            A = torch.cat([A, A_3], dim=1)
            A = torch.cat([Alanl, A], dim=1)
            secret_copy = np.array(secret_copy)
            secret_64 = myresize.Myresize(secret_copy, 64, 64)
            secret_128 = myresize.Myresize(secret_copy, 128, 128)

            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                secret_64 = secret_64.cuda()
                secret_128 = secret_128.cuda()
                A = A.cuda()

            A = Variable(A)
            secret_imgv = Variable(secret_img)
            secret_64v = Variable(secret_64)
            secret_128v = Variable(secret_128)
            cover_imgv = Variable(cover_img)

            container_img = Hnet(A, cover_imgv)
            errH = criterion(container_img, cover_imgv)
            testHlosses.update(errH.item(), batch_size)

            container_img = container_img.detach()

            rev_secret_64, future1 = Rnet1(container_img)
            errR1 = criterion(rev_secret_64, secret_64v)
            testR1losses.update(errR1.item(), batch_size)

            rev_secret_64 = rev_secret_64.detach()
            future1 = future1.detach()

            rev_secret_128, future2 = Rnet2(rev_secret_64, future1)
            errR2 = criterion(rev_secret_128, secret_128v)
            testR2losses.update(errR2.item(), batch_size)

            rev_secret_128 = rev_secret_128.detach()
            future2 = future2.detach()

            rev_secret = Rnet3(rev_secret_128, future2)
            errR3 = criterion(rev_secret, secret_imgv)
            testR3losses.update(errR3.item(), batch_size)

            err_sum = errH + (errR1 + errR2 + errR3) / 3
            testSumLosses.update(err_sum.item(), batch_size)

        test_time = time.time() - start_time
        test_log = "testidation[%d] test_Hloss = %.6f\t test_R1loss = %.6f test_R2loss = %.6f test_R3loss = %.6f\t test_Sumloss = %.6f\t testidation time=%.2f" % (
            epoch, testHlosses.avg, testR1losses.avg,testR2losses.avg,testR3losses.avg, testSumLosses.avg, test_time)
        print_log(test_log, logPath)

        print(
            "#################################################### test end ########################################################")
        return testHlosses, testR1losses,testR2losses,testR3losses, testSumLosses



def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)

    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

def save_result_pic(batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    if not opt.debug:
        showContainer = torch.cat([originalLabelv, ContainerImg], 0)
        showReveal = torch.cat([secretLabelv, RevSecImg], 0)
        resultshowImgName = '%s/ResultshowPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showReveal, resultshowImgName, nrow=batch_size, padding=1, normalize=True)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(showContainer, resultImgName, nrow=batch_size, padding=1, normalize=True)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
