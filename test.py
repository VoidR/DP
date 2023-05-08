import os
import time 
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensor
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Resize
from feat_dist import calculate_recon_error

import resnet_v2 as resnet
from dataset import CIFAR10DatasetDLG
from fl_objs import Server, Client

model_names = ['resnet20', 'resnet32', 'resnet56', 'resnet110', 'resnet1202']

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch, and deep leakage from gradients')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',choices=model_names,help='model architecture: ' + ' | '.join(model_names) +' (default: resnet32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-k', '--clients', default=5, type=int,help='number of clients(default: 5)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=1, type=int,metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--df', dest='differential', action='store_true',help='with differential privacy')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='save_temp', type=str)
parser.add_argument('--index', type=int, default="25", help='the index for leaking images on CIFAR.')
def get_device():
    # 查询GPU显存使用情况，选择空闲显存最大的GPU,如果不存在GPU，则使用CPU
    if torch.cuda.is_available():
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            os.system('rm tmp')
            device = torch.device('cuda:{}'.format(str(np.argmax(memory_gpu))))
        # print("GPU is available")
    else:
        device = torch.device("cpu")
        # print("GPU is not available")
    return device

def criterion(pred, target):
    # print('pred shape',pred.shape)
    # print('target shape',target.shape)
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def base_model():
    return resnet.__dict__[args.arch]().cuda()

def main():
    global args, best_prec1
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    transform_train = Compose([
        Resize(64, 64),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=365,
            p=1.0),
        RandomBrightnessContrast(p=1.0),
        ToTensor()
    ])

    transform_test = Compose([
        Resize(64, 64),
        ToTensor()
    ])

    cudnn.benchmark = True

    batch_size = args.batch_size

    train_data = datasets.CIFAR10(root='~/.torch', train=True, download=True,transform=transform_train)

    s = Server(base_model)
    clients = []
    for i in range(args.clients):
        c_train_dataset = CIFAR10DatasetDLG(dataset=train_data,transform=transform_train, split=(i, args.clients),idx=args.index)
        c_train_loader = torch.utils.data.DataLoader(c_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # print('c_train_loader', len(c_train_loader))
        clients.append(Client(c_train_loader))

    optimizer = torch.optim.Adam(s.current_model.parameters(), args.lr,weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60, 120, 160], gamma=0.1)

    # for epoch in range(0,args.epochs):
    #     print('epoch', epoch,':current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
         
    #     train(train_data_loader, s, clients, criterion, optimizer, epoch)
    #     lr_scheduler.step()

    s.current_model.train()
    s.reset()
    optimizer.zero_grad()
    citers = [c.loader.__iter__() for c in clients]
    for ic, c in enumerate(clients):
        input, target = next(citers[ic])

        input = input.to(0, dtype=torch.float)
        target = target.to(0, dtype=torch.float)

        tt = transforms.ToPILImage()
        filename = os.path.join(args.save_dir,'figs/origin_data.png')
        tt(input[0]).save(filename)

        c.receive_model(s.distribute())
        loss = criterion(c.model(input), target)
        # print("target",target.shape)
        c.local_computation(target, args.differential)
        loss.backward()

        print('本地训练结束，尝试进行深度泄露梯度攻击')
        # 获取客户端原始梯度
        # print("target.size(): ",target.size())
        original_dy_dx = s.get_client_grad(c.model)
        # print('original_dy_dx',original_dy_dx)
        # optim_dlg = args.optim
        # iters_dlg = args.iters
        # dist_dlg = args.dist
        deep_leakage_from_gradients(s.current_model, input.size(), target, original_dy_dx, criterion,ic,args.save_dir,iters_num=900)
        
        s.aggregate(c.model)


def deep_leakage_from_gradients(model, data_size, lable, origin_grad, criterion, ic, save_dir,iters_num,optim='LBFGS',dist='norm'): 
    
    tt = transforms.ToPILImage()
    dummy_data = torch.randn(data_size).cuda().requires_grad_(True)
    dummy_label = torch.randn(lable.size()).cuda().requires_grad_(True)

    if optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[600,900], gamma=0.1)

    elif optim == 'Adam':
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,230,270], gamma=0.1)

    save_step = iters_num//30
    history = []
    min_loss = float("inf")
    min_loss_img = None

    for i in range(iters_num):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data) 
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1))
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            if dist == 'norm':
                # L2范数
                for gx, gy in zip(dummy_grad, origin_grad): 
                    grad_diff += ((gx - gy) ** 2).sum()
            elif dist == 'cosine':
                # 余弦相似度
                 for gx, gy in zip(dummy_grad, origin_grad):
                    grad_diff += (1-(gx * gy).sum() / (gx.norm() * gy.norm()))

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        lr_scheduler.step()

        if i % save_step == 0: 
            current_loss = closure()
            print(i, "current_loss:%.8f" % current_loss.item(),'current lr {:.2e}'.format(optimizer.param_groups[0]['lr']))
            
            history.append(tt(dummy_data[0].cpu()))

            # 保存最小 loss 对应的图像
            if current_loss.item() < min_loss:
                min_loss = current_loss.item()
                min_loss_img = history[-1]
            
        # dummy_data.grad.zero_()
        # dummy_label.grad.zero_()
        
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * save_step))
        plt.axis('off')

    filename = os.path.join(save_dir,'figs/dlg_'+str(ic)+'.png')
    plt.savefig(filename, dpi=300) 

    # 保存最小 loss 对应的图像
    timestamp = int(time.time())
    if args.differential:
        min_loss_filename = os.path.join("path/to/attacked_images/", str(torch.argmax(lable).item())+"/min_loss_{}.png".format(timestamp))
    else:
        min_loss_filename = os.path.join("path/to/df_attacked_images/", str(torch.argmax(lable).item())+"/min_loss_{}.png".format(timestamp))
    min_loss_img.save(min_loss_filename)

    print('feat_dist:',calculate_recon_error(history[-1],dummy_label.argmax().item()))


if __name__ == '__main__':
    main()