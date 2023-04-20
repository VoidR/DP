import torch
import resnet_v2 as resnet
import numpy as np
from dataset import RetinopathyDatasetTrain
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader
import argparse
import torch.backends.cudnn as cudnn
import os
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

model_names = ['resnet20', 'resnet32', 'resnet56', 'resnet110', 'resnet1202']

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', dest='batch_size',default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--enc', dest='encrypt', action='store_true',
                    help='encrypt models')
parser.add_argument('-k', '--clients', default=1, type=int,
                  help='number of clients')
parser.add_argument('--ce', dest='celoss', action='store_true',
                    help='cross entropy loss')
parser.add_argument('--df', dest='differential', action='store_true',
                    help='with differential privacy')
parser.add_argument('--clip', default=-1, type=float,
                  help='gradient clip')
parser.add_argument('--seed', default=0, type=int,
                  help='random seed')

parser.add_argument('--dlg', dest='dlg', action='store_true',
                    help='dlg')

def criterion(y_pred, y_cls):
    c = torch.nn.CrossEntropyLoss()
    return c(y_pred, torch.argmax(y_cls, dim = -1))

def base_model():
    return resnet.__dict__[args.arch]().cuda()

def main():
    global args
    args = parser.parse_args()
    
    # SEED = 0
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

    cudnn.benchmark = True

    batch_size = args.batch_size

    train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta.npy', transform=transform_train, test=args.celoss)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if args.encrypt:
        from fl_objs import Server, Client
    else:
        from fl_objs_v0 import Server, Client

    server = Server(base_model)
    clients = []
    for i in range(args.clients):
        if args.dlg :
            # 如果是dlg，那么每个client只有一张图片
            c_train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta_1.npy', transform=transform_train, split=(i, args.clients), test=args.celoss)
            c_train_loader = DataLoader(c_train_dataset, batch_size=1)
            clients.append(Client(c_train_loader))
        else:
            c_train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta.npy', transform=transform_train, split=(i, args.clients), test=args.celoss)
            c_train_loader = DataLoader(c_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            clients.append(Client(c_train_loader))

    # optimizer = torch.optim.SGD(s.current_model.parameters(), args.lr)
    optimizer = torch.optim.Adam(server.current_model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60, 120, 160], gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        print('epoch {} current lr {:.5e}'.format(epoch,optimizer.param_groups[0]['lr']))
        train(train_loader, server, clients, criterion, optimizer, epoch)
        lr_scheduler.step()

def train(train_loader, server, clients, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    # switch to train mode
    server.current_model.train()

    train_loader = clients[0].loader
    if args.celoss:
        criterion = nn.CrossEntropyLoss()
    for i in range(len(train_loader)):

        server.reset()
        optimizer.zero_grad()

        citers = [c.loader.__iter__() for c in clients]
        for ic, c in enumerate(clients):
            # input, target = next(train_loader)
            input, target = next(citers[ic])
            # input, target = input.cuda(), target.view(-1, 1).cuda()
            input = input.to(0, dtype=torch.float)

            # 如果使用交叉熵损失函数，则将标签转换为LongTensor类型，否则为FloatTensor类型
            if args.celoss:
                target = target.to(0)
            else:
                target = target.to(0, dtype=torch.float)

            # 从服务器接收模型参数
            c.receive_model(server.distribute())
            # 计算损失值
            loss = criterion(c.model(input), target)
            
            if args.encrypt:
                # 若启用加密，在本地计算目标函数
                c.local_computation(target, args.differential)

            if args.dlg and epoch == args.epochs-1:
                # 若开启深度泄露梯度攻击，并且当前为最后一个epoch，则执行攻击
                print('本地训练结束，尝试进行深度泄露梯度攻击')
                # 获取客户端原始梯度
                original_dy_dx = server.get_client_grad(c.model)
                deep_leakage_from_gradients(server.current_model, input.size(), target.size(), original_dy_dx, criterion)   

            # 反向传播计算梯度
            loss.backward()
            # 将模型参数更新到服务器
            server.aggregate(c.model)        

        # 梯度裁剪
        if args.clip > 0.:
            torch.nn.utils.clip_grad_norm_(server.current_model.parameters(), args.clip)

        optimizer.step()
        loss = criterion(server.current_model(input), target)        
        loss.backward()


# def deep_leakage_from_gradients(model, origin_data,lable_size,origin_grad,criterion,ic,save_dir): 

#     tt = transforms.ToPILImage()

#     # print('origin_data',origin_data)
#     # print('origin_data shape',origin_data.shape)

#     # plt.imshow(tt(origin_data[0].cpu()))
#     # filename = os.path.join(save_dir,'figs/data_'+str(ic)+'.png')
#     # tt(origin_data[0]).save(filename)
#     # plt.imsave(filename, tt(origin_data[0].cpu()))
#     # plt.savefig(filename, dpi=300)
    
#     dummy_data = torch.randn(origin_data.size()).cuda().requires_grad_(True)
#     dummy_label =  torch.randn(lable_size).cuda().requires_grad_(True)

#     # optimizer = torch.optim.LBFGS([dummy_data, dummy_label] ,lr=0.1)
#     optimizer = torch.optim.Adam([dummy_data, dummy_label],lr=0.1)
#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 300,500], gamma=0.1)


#     iters_num = 6*100
#     step = 20
#     list_data_diff = []
#     history = []
#     for iters in range(iters_num):
#         def closure():
#             optimizer.zero_grad()
#             dummy_pred = model(dummy_data) 
#             dummy_loss = criterion(dummy_pred, F.softmax(dummy_label, dim=-1)) 
#             dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

#             grad_diff = 0
#             # 欧式距离
#             for gx, gy in zip(dummy_grad, origin_grad): 
#                 grad_diff += ((gx - gy) ** 2).sum()
                
#             # 余弦相似度
#             # for gx, gy in zip(dummy_grad, origin_grad):
#             #     grad_diff += (1-(gx * gy).sum() / (gx.norm() * gy.norm()))

#             grad_diff.backward()
            
#             # 数据间的距离
#             data_diff = 0
#             for gx, gy in zip(dummy_data, origin_data): 
#                 data_diff += ((gx - gy) ** 2).sum()
#             list_data_diff.append(data_diff)

#             return grad_diff

#         optimizer.step(closure)
#         lr_scheduler.step()
#         if iters % step == 0: 
#             current_loss = closure()
#             print(iters, "current_loss:%.4f" % current_loss.item(),"data_diff:%.4f" % list_data_diff[int(step/10)],'current lr {:.2e}'.format(optimizer.param_groups[0]['lr']))
#             history.append(tt(dummy_data[0].cpu()))

#     plt.figure(figsize=(12, 8))
#     for i in range(int(iters_num/step)):
#         plt.subplot(int(iters_num/step/10), 10, i + 1)
#         plt.imshow(history[i])
#         plt.title("iter=%d" % (i * step))
#         plt.axis('off')

#     filename = os.path.join(save_dir,'figs/dlg_'+str(ic)+'.png')
#     plt.savefig(filename, dpi=300)
#     # plt.show()

def deep_leakage_from_gradients(model, origin_data, lable_size, origin_grad, criterion, ic, save_dir,iters_num,optim='LBFGS',dist='norm'): 
    
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

if __name__ == '__main__':
    main()