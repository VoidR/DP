import torch
# from resnet import resnet20
import resnet_v2 as resnet
# import resnet
# from fl_objs import Server, Client
import numpy as np
from dataset import RetinopathyDatasetTrain,CIFAR100DatasetTrain
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import argparse
import torch.backends.cudnn as cudnn
import time 
import os
from torch import nn

from attack_dlg import deep_leakage_from_gradients

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
parser.add_argument('-b', '--batch-size', default=32, type=int,
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
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
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
parser.add_argument('--seed', default=0, type=int,help='random seed')

parser.add_argument('--dlg', dest='dlg', action='store_true',help='dlg')
parser.add_argument('--dataset', dest='dataset', action='store_true',help='used dataset',default='cifar100',type=str)

# def criterion(y_pred, y_cls):
#     return ((y_pred - y_cls)**2).sum(dim=-1).mean() / 2.

def criterion(y_pred, y_cls):
    c = torch.nn.CrossEntropyLoss()
    return c(y_pred, torch.argmax(y_cls, dim = -1))


def base_model():
    # return resnet20().to(0)
    return resnet.__dict__[args.arch]().cuda()

def main():
    global args, best_prec1
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

    transform_test = Compose([
        Resize(64, 64),
        ToTensor()
    ])

    cudnn.benchmark = True


    batch_size = args.batch_size

    # HAM10000数据集
    if args.dataset == 'ham10000':
        train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta.npy', transform=transform_train, test=args.celoss)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/test_meta.npy', transform=transform_test, test=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # CIFAR100
    if args.dataset == 'cifar100':
        dataset = datasets.CIFAR100("~/.torch")
        train_dataset = CIFAR100DatasetTrain(dataset=dataset,transform=transform_train,test=args.celoss)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = CIFAR100DatasetTrain(dataset=dataset,transform=transform_test,test=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.encrypt:
        from fl_objs import Server, Client
    else:
        from fl_objs_v0 import Server, Client

    s = Server(base_model)
    clients = []
    for i in range(args.clients):
        if args.dlg :
            if args.dataset == 'ham10000':
                c_train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta_1.npy', transform=transform_train, split=(i, args.clients), test=args.celoss)
            if args.dataset == 'cifar100':
                c_train_dataset = CIFAR100DatasetTrain(dataset=dataset,transform=transform_train, split=(i, args.clients),test=args.celoss)
        else:
            if args.dataset == 'ham10000':
                c_train_dataset = RetinopathyDatasetTrain(csv_file='./HAM10000/train_meta.npy', transform=transform_train, split=(i, args.clients), test=args.celoss)
        c_train_loader = torch.utils.data.DataLoader(c_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        clients.append(Client(c_train_loader))

    # optimizer = torch.optim.SGD(s.current_model.parameters(), args.lr)
    optimizer = torch.optim.Adam(s.current_model.parameters(), args.lr,
                                    # momentum=args.momentum,
                                    # nesterov=True,
                                    weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 120, 160], gamma=0.1)

    
    # import time
    # start = time.time()

    # for step, batch in enumerate(train_data_loader):
        
    #     inputs = batch["image"]
    #     labels = batch["labels"].view(-1, 1)
    #     x = inputs.to(0, dtype=torch.float)
    #     y = labels.to(0, dtype=torch.float)

    #     s.reset()
    #     opt.zero_grad()

    #     c.receive_model(s.distribute())

    #     loss = criterion(c.model(x), y)
    #     c.local_computation(y)
    #     loss.backward()
    #     s.aggregate(c.model)
    #     opt.step()

    #     loss = criterion(s.current_model(x), y)
    #     # loss.backward()
    #     # opt.step()

    #     print ('Step : ', step, ' Loss: ', loss)

    # end = time.time()
    # print ('Eduration : ', end - start)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            s.current_model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, s.current_model, criterion)
        exit()

    best_prec1 = -1e8
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, s, clients, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, s.current_model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': s.current_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': s.current_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, server, clients, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    server.current_model.train()
    
    end = time.time()
    
    train_loader = clients[0].loader
    # print ('!!!!', len(clients[1].loader))
    # for i, (input, target) in enumerate(train_loader):
    if args.celoss:
        criterion = nn.CrossEntropyLoss()
    for i in range(len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # input = input.cuda()
        # target = target.cuda()

        # compute output
        # output = model(input)
        # loss = criterion(output, target)

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        server.reset()
        optimizer.zero_grad()

        citers = [c.loader.__iter__() for c in clients]
        for ic, c in enumerate(clients):
            # input, target = next(train_loader)
            input, target = next(citers[ic])
            # input, target = input.cuda(), target.view(-1, 1).cuda()
            input = input.to(0, dtype=torch.float)
            if args.celoss:
                target = target.to(0)
            else:
                target = target.to(0, dtype=torch.float)

            if args.encrypt:
                c.receive_model(server.distribute())
                loss = criterion(c.model(input), target)
                c.local_computation(target, args.differential)
                loss.backward()
                server.aggregate(c.model)
            else:
                c.receive_model(server.distribute())
                loss = criterion(c.model(input), target)
                loss.backward()
                server.aggregate(c.model)
            
            if args.dlg and epoch == args.epochs-1:
                # 若开启深度泄露梯度攻击，并且当前为最后一个epoch，则执行攻击
                print('本地训练结束，尝试进行深度泄露梯度攻击')
                # 获取客户端原始梯度
                original_dy_dx = server.get_client_grad(c.model)
                deep_leakage_from_gradients(server.current_model, input, target.size(), original_dy_dx, criterion,ic,args.save_dir)

        if args.clip > 0.:
            torch.nn.utils.clip_grad_norm_(server.current_model.parameters(), args.clip)

        optimizer.step()
        loss = criterion(server.current_model(input), target)        
        loss.backward()
        # output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        # prec1 = accuracy(output.data, target)[0]
        prec1 = loss.data
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    crt = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda(async=True)
            # input_var = torch.autograd.Variable(input, volatile=True).cuda()
            # target_var = torch.autograd.Variable(target, volatile=True)
            input = input.to(0, dtype=torch.float)
            target = target.to(0)
            # print (target)

            # if args.half:
            #     input_var = input_var.half()

            # compute output
            output = model(input)
            loss = crt(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()