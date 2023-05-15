import os
import time 
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensor
from albumentations import Compose, Resize , Normalize, HorizontalFlip

import resnet_v2 as resnet
from dataset import CIFAR10DatasetDLG

model_names = ['resnet20', 'resnet32', 'resnet56', 'resnet110', 'resnet1202']

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch, and deep leakage from gradients')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',choices=model_names,help='model architecture: ' + ' | '.join(model_names) +' (default: resnet20)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('-k', '--clients', default=5, type=int,help='number of clients(default: 5)')
parser.add_argument('--seed', default=3407, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=1, type=int,metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--df', dest='differential', action='store_true',help='with differential privacy')
parser.add_argument('--save-dir', dest='save_dir',help='The directory used to save the trained models',default='save_temp', type=str)
parser.add_argument('--save-figs-dir', dest='save_figs_dir',help='The directory used to save the trained models',default='save_temp', type=str)
parser.add_argument('--index', type=int, default="369", help='the index for leaking images on CIFAR.')
parser.add_argument('--iters', type=int, default="4800", help='the iters for leaking images on CIFAR.')
parser.add_argument('--enc', dest='encrypt', action='store_true',help='encrypt models')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    memory_gpu = [torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in range(gpu_count)]
    device = torch.device('cuda:{}'.format(str(memory_gpu.index(max(memory_gpu)))))


def criterion(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def base_model():
    return resnet.__dict__[args.arch]().to(device)


def main():
    global args, best_prec1
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # transform_train = Compose([
    #     Resize(32, 32),
    #     ShiftScaleRotate(
    #         shift_limit=0.1,
    #         scale_limit=0.1,
    #         rotate_limit=365,
    #         p=1.0),
    #     RandomBrightnessContrast(p=1.0),
    #     ToTensor()
    # ])

    # norm_mean = [0.485, 0.456, 0.406]
    # norm_std = [0.229, 0.224, 0.225]
    transform_train = Compose([
        Resize(32, 32),
        HorizontalFlip(p=0.5),
        # Normalize(norm_mean, norm_std),
        ToTensor()
    ])

    cudnn.benchmark = True
    batch_size = args.batch_size

    train_data = datasets.CIFAR10(root='~/.torch', train=True, download=True,transform=transform_train)

    if args.encrypt:
        from fl_objs import Server, Client
    else:
        from fl_objs_v0 import Server, Client

    server = Server(base_model)
    clients = []
    for i in range(args.clients):
        c_train_dataset = CIFAR10DatasetDLG(dataset=train_data,transform=transform_train, split=(i, args.clients),idx=args.index)
        c_train_loader = torch.utils.data.DataLoader(c_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        clients.append(Client(c_train_loader))

    server.current_model.train()
    server.reset()
    citers = [c.loader.__iter__() for c in clients]
    for ic, c in enumerate(clients):
        input, target = next(citers[ic])

        input = input.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        tt = transforms.ToPILImage()
        filename = os.path.join(args.save_dir,'figs/origin_data.png')
        tt(input[0]).save(filename)

        c.receive_model(server.distribute())
        loss = criterion(c.model(input), target)
        target = torch.squeeze(target, dim=1)
        if args.encrypt:
            c.local_computation(target, args.differential)
        loss.backward()

        print('本地训练结束，尝试进行深度泄露梯度攻击')
        # 获取客户端原始梯度
        original_dy_dx = server.get_client_grad(c.model)

        # optim_dlg = args.optim
        deep_leakage_from_gradients(server.current_model, input, target, original_dy_dx, criterion,ic,max_iterations=args.iters)
        
        server.aggregate(c.model)


def deep_leakage_from_gradients(model, input, label, origin_grad, criterion, ic, max_iterations,optim='Adam'): 
    start_time = time.time()
    
    tt = transforms.ToPILImage()
    dummy_data = torch.randn(input.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(label.size()).to(device).requires_grad_(True)

    if optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 180, 240], gamma=0.1)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[max_iterations // 2.667, max_iterations // 1.6,max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 900, 1400], gamma=0.1)

    save_step = max_iterations // 30
    history = []
    min_loss = float("inf")
    min_loss_img = None

    for iters in range(max_iterations):
        closure = _gradient_closure(model, optimizer, dummy_data, origin_grad, dummy_label, criterion)
        rec_loss = optimizer.step(closure)
        scheduler.step()

        if (iters + 1 == max_iterations) or iters % 500 == 0:
            # print(f'It: {iters}. Rec. loss: {rec_loss.item():2.4f}. Lr: {optimizer.param_groups[0]['lr']}', flush=True)
            print(f'It: {iters}. Rec. loss: {rec_loss.item():2.4f}. Lr: {optimizer.param_groups[0]["lr"]:.2e}', flush=True)

            # history.append(dummy_data.detach_())

    print(f'Total time: {time.time()-start_time}.')

    # 保存图片
    save_image(dummy_data.cpu().clone(), '{}/figs/rec_{}.png'.format(args.save_dir, args.index))
    save_image(input.cpu().clone(), '{}/figs/ori_{}.png'.format(args.save_dir, args.index))


    #     optimizer.zero_grad()
    #     dummy_pred = model(dummy_data) 
    #     dummy_onehot_label = F.softmax(dummy_label, dim=-1)
    #     dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
    #     dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

    #     grad_diff = 0
    #     # for gx, gy in zip(dummy_dy_dx, origin_grad):
    #     #     grad_diff += ((gx - gy) ** 2).sum()

    #     for gx, gy in zip(dummy_dy_dx, origin_grad):
    #             grad_diff += (1 - (gx * gy).sum() / (gx.norm() * gy.norm()))
        
    #     grad_diff.backward()
    #     optimizer.step()
    #     scheduler.step()

    #     if iters % save_step == 0: 
    #         current_loss = grad_diff
    #         print(iters, "current_loss:%.8f" % current_loss.item(), 'current lr {:.2e}'.format(optimizer.param_groups[0]['lr']))
    #         history.append(tt(dummy_data[0].cpu()))

    #         # 保存最小 loss 对应的图像
    #         if current_loss.item() < min_loss:
    #             min_loss = current_loss.item()
    #             min_loss_img = history[-1]

    # plt.figure(figsize=(12, 8))
    # for i in range(30):
    #     plt.subplot(3, 10, i + 1)
    #     plt.imshow(history[i])
    #     plt.title("iter=%d" % (i * save_step))
    #     plt.axis('off')

    # filename = os.path.join(args.save_dir,'figs/dlg_'+str(ic)+'.png')
    # plt.savefig(filename, dpi=300) 

    # # 保存最小 loss 对应的图像
    # timestamp = int(time.time())
    # min_loss_filename = os.path.join(args.save_figs_dir, str(torch.argmax(label).item())+"/min_loss_{}.png".format(timestamp))
    # min_loss_img.save(min_loss_filename)


def _gradient_closure(model, optimizer, x_trial, input_gradient, label, loss_fn):

    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        loss = loss_fn(model(x_trial), label)
        param_list = [param for param in model.parameters() if param.requires_grad]
        gradient = torch.autograd.grad(loss, param_list, create_graph=True)
        rec_loss = reconstruction_costs([gradient], input_gradient, cost_fn='l2', indices='def',weights='equal')
        
        rec_loss += 0.0001 * total_variation(x_trial)
        rec_loss.backward()

        x_trial.grad.sign_()
        
        return rec_loss
    return closure


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))
    cnt = 0
    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
                cnt = 1
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
                cnt = 1
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
                cnt = 1
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
                cnt = 1
            elif cost_fn == 'out_sim':
                if len(trial_gradient[i].shape) >= 2:
                    for j in range(trial_gradient[i].shape[0]):
                        costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i][j].flatten(),
                                                                   input_gradient[i][j].flatten(),
                                                                   0, 1e-10) * weights[i]
                        cnt += 1
                else:
                    costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
                    cnt += 1
                # print(sim_out.item())
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            # costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
            norm = max(pnorm[0].sqrt() * pnorm[1].sqrt(), torch.tensor(1e-8, device=pnorm[0].device))
            costs = 1 + costs / norm
        # Accumulate final costs
        total_costs += costs / cnt
    # print(type(gradients), len(gradients), type(gradients[0]), type(total_costs))
    if torch.isinf(total_costs):
        print('inf', cost_fn, costs, cnt)
        exit(0)
    if torch.isnan(total_costs):
        print('nan', cost_fn, costs, cnt)
        exit(0)
    # exit(0)
    
    return total_costs / len(gradients)


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

if __name__ == '__main__':
    main()