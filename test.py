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

import resnet_v2 as resnet
from dataset import CIFAR10DatasetTrain
from fl_objs import Server, Client

model_names = ['resnet20', 'resnet32', 'resnet56', 'resnet110', 'resnet1202']

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch, and deep leakage from gradients')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',choices=model_names,help='model architecture: ' + ' | '.join(model_names) +' (default: resnet32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-k', '--clients', default=1, type=int,help='number of clients')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--df', dest='differential', action='store_true',help='with differential privacy')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='save_temp', type=str)

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
    test_data = datasets.CIFAR10(root='~/.torch', train=False,transform=transform_test)

    train_data_loader=DataLoader(train_data,batch_size=batch_size)
    test_data_loader=DataLoader(test_data,batch_size=batch_size)

    print('train_data_loader', len(train_data_loader))


    s = Server(base_model)
    clients = []
    for i in range(args.clients):
        c_train_dataset = CIFAR10DatasetTrain(dataset=train_data,transform=transform_train, split=(i, args.clients),idx=30)
        c_train_loader = torch.utils.data.DataLoader(c_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        print('c_train_loader', len(c_train_loader))
        clients.append(Client(c_train_loader))

    optimizer = torch.optim.Adam(s.current_model.parameters(), args.lr,weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60, 120, 160], gamma=0.1)

    for epoch in range(1,args.epochs):
        print('epoch', epoch,':current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
         
        train(train_data_loader, s, clients, criterion, optimizer, epoch)
        lr_scheduler.step()

def train(train_loader, server, clients, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    server.current_model.train()

    end = time.time()

    server.reset()
    optimizer.zero_grad()

    citers = [c.loader.__iter__() for c in clients]
    for ic, c in enumerate(clients):
        input, target = next(citers[ic])

        input = input.to(0, dtype=torch.float)
        target = target.to(0, dtype=torch.float)

        tt = transforms.ToPILImage()
        filename = os.path.join(args.save_dir,'figs/origin_data.png')
        tt(input[0]).save(filename)

        c.receive_model(server.distribute())
        loss = criterion(c.model(input), target)
        c.local_computation(target, args.differential)
        # loss.backward()

        print('本地训练结束，尝试进行深度泄露梯度攻击')
        # 获取客户端原始梯度
        
        original_dy_dx = server.get_client_grad(c.model)
        # optim_dlg = args.optim
        # iters_dlg = args.iters
        # dist_dlg = args.dist
        deep_leakage_from_gradients(server.current_model, input, target.size(), original_dy_dx, criterion,ic,args.save_dir,iters_num=600)
        
        server.aggregate(c.model)

def deep_leakage_from_gradients(model, origin_data, lable_size, origin_grad, criterion, ic, save_dir,iters_num,optim='LBFGS',dist='cosine'): 
    
    tt = transforms.ToPILImage()
    dummy_data = torch.randn(origin_data.size()).cuda().requires_grad_(True)
    dummy_label = torch.randn(lable_size).cuda().requires_grad_(True)

    if optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,300], gamma=0.1)

    elif optim == 'Adam':
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,300,500], gamma=0.1)

    iters_num = iters_num
    save_step = iters_num//30
    history = []
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
            
        dummy_data.grad.zero_()
        dummy_label.grad.zero_()
        
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * save_step))
        plt.axis('off')

    filename = os.path.join(save_dir,'figs/dlg_'+str(ic)+'.png')
    plt.savefig(filename, dpi=300)  



class AverageMeter(object):
    """Computes and stores the average and current value"""
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