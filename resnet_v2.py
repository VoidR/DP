'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import utils

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    # if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        # init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _post(y_logit, y_gt, y_last, global_var, y_current, weight):
    batch_size = y_last.size(0)

    a, b, alpha, cls_weight = global_var

    y_last = y_last.view(batch_size, -1)
    _y_cur = y_current.view(batch_size, -1)
    delta_y = y_logit - y_gt
    delta_y = delta_y.view(batch_size, -1)
    y_logit, y_gt = y_logit.view(batch_size, -1), y_gt.view(batch_size, -1)
    
    # Q_sum = torch.autograd.grad(y_last, y_current, torch.ones_like(y_last), retain_graph=True)[0].view(batch_size, -1)
    
    sigma_conv_a = torch.autograd.grad(y_logit, weight, torch.einsum('c,k->kc', a, alpha), retain_graph=True)[0]
    sigma_conv_a += torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, a), torch.ones_like(y_last)), retain_graph=True)[0]

    sigma_conv_b = torch.autograd.grad(y_logit, weight, torch.einsum('c,k->kc', b, alpha), retain_graph=True)[0]
    sigma_conv_b += torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, b), torch.ones_like(y_last)), retain_graph=True)[0]

    beta = torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', alpha, torch.ones_like(y_last)), retain_graph=True)[0]

    return sigma_conv_a / batch_size, sigma_conv_b / batch_size, beta / batch_size

def rand(size, num_k=3, abs=True):
    mode = 1

    if mode == 0:
        rv = torch.rand(size)
    else:
        rv = torch.Tensor(utils.generate_normal(size, num_k))
        if abs:
            rv = rv.abs()
    return rv

class Linear(nn.Module):

    def __init__(self, num_in, num_out):
        super(Linear, self).__init__()
        self.fc = nn.Linear(num_in, num_out, bias=False)
        self.num_in, self.num_out = num_in, num_out
    def forward(self, x):
        self.y = self.fc(x)
        return self.y

    # def randomize(self, last_r = None, set_r = None):

    #     if set_r is None:
    #         self.rc1 = torch.ones(self.fc.weight.size(0)).to(0) + 1e-5
    #     else:
    #         self.rc1 = set_r
        
    #     if last_r is None:
    #         self.r1 = torch.einsum('i,j->ij', self.rc1, torch.ones(self.fc.weight.size(1)).to(0))
    #     else:
    #         self.r1 = torch.einsum('i,j->ij', self.rc1, 1. / last_r.repeat(int(self.num_in / last_r.size(0)), 1).t().reshape(-1))

    #     self.a, self.b, self.gamma = torch.rand(self.num_out).to(0) + 1e-5, torch.rand(self.num_out).to(0) + 1e-5, torch.rand(2).to(0)  + 1e-5
    #     self.gamma[1] = 0
    #     self.rcls = self.a * self.gamma[0] + self.b * self.gamma[1]
    #     self.v = torch.sum(self.rcls ** 2)

    #     self.fc.weight = torch.nn.Parameter(self.r1 * self.fc.weight + self.rcls.view(-1, 1))

    #     return self.a, self.b, self.gamma, self.v

    def randomize(self, last_r = None, set_r = None):

        if set_r is None:
            self.rc = torch.rand(self.fc.weight.size(0)).to(0)
            # if last_r is None:
            #     self.rc = torch.rand(self.out_channel).to(0) * 0.5 + 1.5
            # else:
            #     assert (2 * last_r.min().item() > last_r.max().item())
            #     self.rc = torch.rand(self.out_channel).to(0) * (2 * last_r.min() - last_r.max()) + last_r.max()
        else:
            self.rc = set_r

        if last_r is None:
            self.r = torch.einsum('i,j->ij', self.rc, torch.ones(self.fc.weight.size(1)).to(0))
        else:
            self.r = torch.einsum('i,j->ij', self.rc, 1. / last_r.repeat(int(self.num_in / last_r.size(0)), 1).t().reshape(-1))
        # print ('r last : ', self.r.max(), self.r.min())
        # if self.last:
        self.a, self.b, self.gamma = torch.rand(self.num_out).to(0) * 1e-1, torch.zeros(self.num_out).to(0), torch.rand(2).to(0)  

        self.rcls = self.a * self.gamma[0] + self.b * self.gamma[1]

        self.v = torch.sum(self.rcls ** 2)
        self.fc.weight = torch.nn.Parameter(self.r * self.fc.weight + self.rcls.view(-1, 1))

        return self.a, self.b, self.gamma, self.v
        # else:
        #     self.fc.weight = torch.nn.Parameter(self.r * self.fc.weight)            
        #     return self.rc

    def post(self, y_logit, y_gt, y_last, global_var):

        a, b, alpha, cls_weight = global_var
        batch_size = y_last.size(0)
        delta_y = (y_logit - y_gt).view(batch_size, -1)
        y_last = y_last.view(batch_size, -1)
        sigma_conv = torch.einsum('kj,k->kj', y_last, alpha)
        sigma_conv_a = torch.einsum('i,kj->kij', a, sigma_conv).sum(dim=0)
        sigma_conv_b = torch.einsum('i,kj->kij', b, sigma_conv).sum(dim=0)

        self.post_data = sigma_conv_a / batch_size, sigma_conv_b / batch_size
        

    # def correction(self, gamma, v, post_data, grad, r):

    #     sigma_conv_a, sigma_conv_b = post_data
    #     sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
        
    #     delta_conv1 = sigma_conv

    #     # self.fc.weight.grad = (self.fc.weight.grad - self.delta_conv1) * self.r1
    #     return (grad - delta_conv1) * r

    def correction(self, gamma, v, post_data, grad, r):
        sigma_conv_a, sigma_conv_b, beta = post_data

        return (grad - gamma[0] * sigma_conv_a - gamma[1] * beta + gamma[0] * gamma[1] * sigma_conv_b) * r

    def get_grad(self):
        return self.fc.weight.grad

    def get_rectify_grad(self):
        return self.rectify_grad

    def set_grad(self, grad):
        self.fc.weight.grad = grad

    def get_r(self):
        return self.r

    # def get_appro_norm(self):
    #     return self.rectify_grad.abs().max() + self.sigma_conv_a.abs().max() + self.sigma_conv_b.abs().max() + self.beta().abs().max()

    def aggregate_grad(self, grad, counter):
        try:
            self.fc.weight.grad = (self.fc.weight.grad * (counter - 1) + grad) / counter
        except Exception as e:
            self.fc.weight.grad = grad

    def update(self, lr):
        self.fc.weight = nn.Parameter(self.fc.weight - self.fc.weight.grad * lr)

class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.id_filter = torch.randn(out_channel,out_channel,1,1).to(0); self.id_filter[:, :, 0, 0] = torch.eye(out_channel).to(0)
        # print (self.id_filter)
        self.in_channel, self.out_channel = in_channel, out_channel
        
        self.rc, self.r0 = 1., 1.

    def forward(self, x):
        self.x0 = x
        self.y = self.conv(x)
        self.y0 = self.y
        self.y = F.conv2d(self.y, self.id_filter, padding=0)
        
        # print ('Zero : ', ((self.y - y0)**2).sum())
        # try:
        #     print (y0.size(), self.y.size(), self.in_channel, self.out_channel, self.rc.size(), self.r0.size())
        #     print ('WTF : ',  ((self.rc.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * y0 / self.r0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - self.y)**2).sum())
        # except Exception as e:
        #     pass
        
        
        return self.y

    def randomize(self, last_r = None, set_r = None):
        if last_r is None:
            # self.rc = 1e-2 * (torch.rand(self.out_channel).to(0) * 0.5 + 1.5)
            rc0 = rand(self.out_channel, 2).to(0) * 1e-2 + 1e-8
        else:
            rc0 = rand(self.out_channel, 3).to(0) * 1e-2 + 1e-8

        if set_r is None:
            # rc1 = torch.rand(self.out_channel).to(0) + 1e-8
            rc1 = 1. / rand(self.out_channel, 3).to(0) * 1e-2 + 1e-8
            # rc1 = rand(self.out_channel, 3).to(0) + 1e-8
            # assert (2 * last_r.min().item() > last_r.max().item())
            # self.rc = torch.rand(self.out_channel).to(0) * (2 * last_r.min() - last_r.max()) + last_r.max()
            # print (self.rc.max(), self.rc.min())
            # assert()
        else:
            rc1 = set_r
        # rc0 = rc1
        if last_r is None:
            self.r = torch.einsum('i,j->ij', rc0, torch.ones(self.in_channel).to(0))
        else:
            self.r = torch.einsum('i,j->ij', rc0, 1. / last_r)

        self.id_filter = torch.randn(self.out_channel,self.out_channel,1,1).to(0); self.id_filter[:, :, 0, 0] = torch.eye(self.out_channel).to(0)
        # print (self.id_filter)
        self.id_filter = torch.einsum('i,j->ij', rc1, 1. / rc0).unsqueeze(-1).unsqueeze(-1) * self.id_filter
        self.id_filter = torch.autograd.Variable(self.id_filter, requires_grad=False).to(0)
        # print ('r conv : ', self.r.max(), self.r.min())
        # if last_r is not None:
        #     print (last_r.size(), self.r.size(), self.conv.weight.size())
        self.conv.weight = torch.nn.Parameter(self.conv.weight * self.r.unsqueeze(-1).unsqueeze(-1))
        
        # print ('Conv max min', rc1.max(), rc1.min())

        self.r0 = rc0
        self.rc = rc1
        
        return self.rc

    def post(self, y_logit, y_gt, y_last, global_var):

        self.post_data = _post(y_logit, y_gt, y_last, global_var, self.y, self.conv.weight)
        

    # def correction(self, gamma, v, post_data, grad, r):

    #     sigma_conv_a, sigma_conv_b, beta = post_data
    #     sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
    #     beta = beta * v
    #     delta_conv1 = sigma_conv - beta

    #     # print (self.conv1.weight.grad.size(), self.r1.size())
    #     # self.conv.weight.grad = (self.conv.weight.grad - self.delta_conv1) * self.r.unsqueeze(-1).unsqueeze(-1)
    #     return (grad - delta_conv1) * r.unsqueeze(-1).unsqueeze(-1)

    def correction(self, gamma, v, post_data, grad, r):
        sigma_conv_a, sigma_conv_b, beta = post_data

        # print (((grad - gamma[0] * sigma_conv_a - gamma[1] * beta + gamma[0] * gamma[1] * sigma_conv_b) * r.unsqueeze(-1).unsqueeze(-1)).norm())
        # print ('What is the grad : ',  grad.norm(), sigma_conv_a.norm(), sigma_conv_b.norm(), beta.norm(), ((grad - gamma[0] * sigma_conv_a - gamma[1] * beta + gamma[0] * gamma[1] * sigma_conv_b) * r.unsqueeze(-1).unsqueeze(-1)).norm())
        return (grad - gamma[0] * sigma_conv_a - gamma[1] * beta + gamma[0] * gamma[1] * sigma_conv_b) * r.unsqueeze(-1).unsqueeze(-1)


    def get_grad(self):
        return self.conv.weight.grad

    def get_rectify_grad(self):
        return self.rectify_grad

    def set_grad(self, grad):
        self.conv.weight.grad = grad

    def get_r(self):
        return self.r

    def aggregate_grad(self, grad, counter):
        try:
            self.conv.weight.grad = (self.conv.weight.grad * (counter - 1) + grad) / counter
        except Exception as e:
            self.conv.weight.grad = grad

    def update(self, lr):
        self.conv.weight = nn.Parameter(self.conv.weight - self.conv.weight.grad * lr)

        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.Sequential()
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     # nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.o1 = out
        out = self.bn2(self.conv2(out))
        self.o2 = out
        out += self.shortcut(x)
        self.s = out
        out = F.relu(out)
        self.out = out
        return out

    def randomize(self, last_r):
        self.r1 = self.conv1.randomize(last_r)
        
        if len(self.shortcut) > 0:
            self.r2 = self.conv2.randomize(self.r1)
            # print ('SHORT-CUT in')
            self.shortcut[0].randomize(last_r, self.r2)
            # print ('SHORT-CUT out')
        else:
            # print ('REALLY in')
            self.r2 = self.conv2.randomize(self.r1, last_r)
            # print ('REALLY out')
        return self.r2

    def post(self, y_logit, y_gt, y_last, global_var):
        self.conv1.post(y_logit, y_gt, y_last, global_var)
        self.conv2.post(y_logit, y_gt, y_last, global_var)
        if len(self.shortcut) > 0:
            self.shortcut[0].post(y_logit, y_gt, y_last, global_var)

    def correction(self, gamma, v):
        self.conv1.correction(gamma, v)
        self.conv2.correction(gamma, v)
        if len(self.shortcut) > 0:
            self.shortcut[0].correction(gamma, v)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.cls_num = num_classes
        self.in_planes = 16

        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        self.y0 = out
        out = self.layer1(out)
        self.y1 = out
        out = self.layer2(out)
        self.y2 = out
        out = self.layer3(out)
        self.y3 = out
        out = F.avg_pool2d(out, out.size()[3])
        self.p = out
        out = out.view(out.size(0), -1)
        self.y_last = out
        out = self.linear(out)
        self.logits = out
        return out

    def randomize(self):
        self.r1 = self.conv1.randomize()
        r = self.r1
        
        for i in range(len(self.layer1)):
            r = self.layer1[i].randomize(r)
        self.r2 = r
        for i in range(len(self.layer2)):
            # print (i)
            r = self.layer2[i].randomize(r)
        self.r3 = r
        for i in range(len(self.layer3)):
            r = self.layer3[i].randomize(r)
        self.r4 = r

        self.a, self.b, self.gamma, self.v = self.linear.randomize(self.r4, torch.ones(self.linear.fc.weight.size(0)).to(0))

        self.rcls = self.linear.rcls
        self.q = torch.rand(self.cls_num).to(0)

        # total_random = 0
        # for m in self.named_modules():
        #     if type(m[1]) is Linear or type(m[1]) is Conv2d:
        #         print (m[1].rc)
        #         total_random += len(m[1].rc)
        # print ('Total Random Numbers : ', total_random)


    # def post(self, y_gt):
    #     batch_size = y_gt.size(0)

    #     delta_y = self.logits - y_gt
    #     cls_weight = self.linear.fc.weight
    #     self.alpha = self.y_last.view(batch_size, -1).sum(dim=-1)

    #     global_var = (self.a, self.b, self.alpha, cls_weight)
    #     opt = torch.optim.SGD(self.parameters(), 0)

    #     # self.conv1.post(self.logits, y_gt, self.y_last, global_var)
    #     # for i in range(len(self.layer1)):
    #     #     self.layer1[i].post(self.logits, y_gt, self.y_last, global_var)
    #     # for i in range(len(self.layer2)):
    #     #     self.layer2[i].post(self.logits, y_gt, self.y_last, global_var)
    #     # for i in range(len(self.layer3)):
    #     #     self.layer3[i].post(self.logits, y_gt, self.y_last, global_var)

    #     logits = self.logits.view(batch_size, -1)
    #     y_last = self.y_last.view(batch_size, -1)
    #     delta_y = delta_y.view(batch_size, -1)

    #     opt.zero_grad()
    #     # sig_1 = torch.einsum('kc,c->k', torch.einsum('c,k->kc', self.a, self.alpha), self.logits.view(batch_size, -1))
    #     logits.backward(torch.einsum('c,k->kc', self.a, self.alpha), retain_graph=True)
    #     for m in self.named_modules():
    #         if type(m[1]) is Conv2d:
    #             m[1].sigma_conv_a = m[1].conv.weight.grad / batch_size
    #             m[1].sigma_conv_b = 0

    #     opt.zero_grad()
    #     y_last.backward(torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, self.a), torch.ones_like(y_last)), retain_graph=True)
    #     for m in self.named_modules():
    #         if type(m[1]) is Conv2d:
    #             m[1].sigma_conv_a += (m[1].conv.weight.grad / batch_size)

    #     opt.zero_grad()
    #     y_last.backward(torch.einsum('k,kl->kl', self.alpha, torch.ones_like(y_last)), retain_graph=True)
    #     for m in self.named_modules():
    #         if type(m[1]) is Conv2d:
    #             m[1].beta = m[1].conv.weight.grad / batch_size
    #             m[1].post_data = (m[1].sigma_conv_a, 0, m[1].beta)
    #     opt.zero_grad()

    #     self.linear.post(self.logits, y_gt, self.y_last, global_var)

    def correction(self):

        self.conv1.correction(self.gamma, self.v)
        for i in range(len(self.layer1)):
            self.layer1[i].correction(self.gamma, self.v)
        for i in range(len(self.layer2)):
            self.layer2[i].correction(self.gamma, self.v)
        for i in range(len(self.layer3)):
            self.layer3[i].correction(self.gamma, self.v)
        self.linear.correction(self.gamma, self.v)

        # self.linear.y -= torch.einsum('c,k->kc', self.linear.rcls, self.alpha).view(self.linear.y.size())
        # self.conv1.y = torch.einsum('kchw,c->kchw', self.conv1.y, 1. /self.conv1.rc)

    def post_temp(self, grad_L2ylast):

        batch_size = self.y_last.size(0)

        self.alpha = self.y_last.view(batch_size, -1).sum(dim=-1)
        global_var = self.a, self.b, self.alpha, None


        opt = torch.optim.SGD(self.parameters(), 0)

        logits = self.logits.view(batch_size, -1)
        y_last = self.y_last.view(batch_size, -1)
        
        # print("logits shape",logits.shape)
        # print("y_last shape",y_last.shape)
        # print("grad_L2ylast shape",grad_L2ylast.shape)

        opt.zero_grad()
        # print ('Let see : ', grad_L2ylast.norm())
        logits.backward(grad_L2ylast, retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear or type(m[1]) is Conv2d:
                m[1].rectify_grad = m[1].get_grad() / batch_size

        opt.zero_grad()
        y_last.backward(torch.einsum('k,kl->kl', torch.einsum('kc,c->k', grad_L2ylast, self.a), torch.ones_like(y_last)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear or type(m[1]) is Conv2d:
                m[1].sigma_conv_a = (m[1].get_grad() / batch_size)
                # m[1].sigma_conv_b = 0

        opt.zero_grad()
        # y_last.backward(torch.einsum('k,kl->kl', torch.einsum('kc,c->k', self.q.unsqueeze(0).repeat(batch_size, 1), self.a), torch.ones_like(y_last)), retain_graph=True)
        y_last.backward((self.q * self.a).sum() * torch.ones_like(y_last), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear or type(m[1]) is Conv2d:
                m[1].sigma_conv_b = (m[1].get_grad() / batch_size)


        opt.zero_grad()

        logits.backward(torch.einsum('c,k->kc', self.q, torch.ones(batch_size).to(0)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear or type(m[1]) is Conv2d:
                m[1].beta = m[1].get_grad() / batch_size

                m[1].post_data = (m[1].sigma_conv_a, m[1].sigma_conv_b, m[1].beta)

        opt.zero_grad()

    def correction_temp(self, gamma):
        for m in self.named_modules():
            if type(m[1]) is Linear:
                m[1].fc.weight.grad = (m[1].rectify_grad - gamma[0] * m[1].sigma_conv_a - gamma[1] * m[1].beta + gamma[1] * gamma[0] * m[1].sigma_conv_b) * m[1].r
            elif type(m[1]) is Conv2d:
                m[1].conv.weight.grad = (m[1].rectify_grad - gamma[0] * m[1].sigma_conv_a - gamma[1] * m[1].beta + gamma[1] * gamma[0] * m[1].sigma_conv_b) * m[1].r.unsqueeze(-1).unsqueeze(-1)

    def interaction_1_c2s(self):
        # Additional Round 1: Client -> Server

        logits = self.logits
        batch_size, cls_num = logits.size()
        logits_exp = torch.exp(logits)
        # print ('logits ? ', logits.norm(), logits.max(), logits.min(), torch.einsum('kl,kj->klj', logits_exp, 1. / logits_exp).permute(0,2,1).norm())
        # if logits.norm() < 1e-5:
        # print ('norms : ', self.y0.norm(), self.y1.norm(), self.y2.norm(), self.y3.norm(), self.p.norm(), self.y_last.norm())
            # for i in range(len(self.layer3)):
            #     print (i, '-th : ', self.layer3[0].o1.norm(), self.layer3[0].o2.norm(), self.layer3[0].s.norm(), self.layer3[0].out.norm())
        # print ('See : ', self.layer3[0].conv1.conv.weight.data.norm(), self.layer3[0].conv1.r0.norm(), self.layer3[0].conv1.r0.max(), self.layer3[0].conv1.r0.min(), self.layer3[0].conv1.x0.norm(), self.layer3[0].conv1.y0.norm(), self.layer3[0].conv1.y.norm())
        self.i1c_lambda = torch.zeros(batch_size, cls_num).to(0) 
        self.i1c_mu = self.i1c_lambda.unsqueeze(-1) + torch.einsum('kl,kj->klj', logits_exp, 1. / logits_exp).permute(0,2,1)
        
        mask = torch.diag(torch.ones(cls_num)).to(0)
        self.i1c_mu = self.i1c_mu * (1 - mask)
        self.alpha = self.y_last.view(batch_size, -1).sum(dim=-1)

        # return logits, self.alpha
        return self.i1c_mu, self.alpha

    # def interaction_1_s2c(self, recv):
    #     # Addtional Round 2: Server -> Client

    #     i1c_mu, alpha = recv
    #     batch_size, cls_num, _ = i1c_mu.size()

    #     # diff = torch.einsum('k,c->kc', alpha, self.rcls)

    #     # pred_recover = torch.exp(-diff) * i1c_mu
    #     # pred_recover = torch.exp(-diff + i1c_mu)
    #     # # print ('wtf : ', pred_recover.norm(), pred_recover.sum(dim=-1))
    #     # pred_recover = pred_recover / pred_recover.sum(dim=-1).unsqueeze(-1)
    #     # # print ('original : ', pred_recover.norm())
    #     # return pred_recover + self.q.unsqueeze(0) * self.gamma[1], None
    #     self.i1s_delta = torch.zeros(cls_num).to(0) 
    #     diff = self.rcls.unsqueeze(0).repeat(cls_num, 1) - self.rcls.unsqueeze(-1)
        
    #     self.i1s_r = self.i1s_delta.unsqueeze(0).unsqueeze(-1) - diff.unsqueeze(0) * alpha.unsqueeze(-1).unsqueeze(-1)
        
    #     mask = torch.diag(torch.ones(cls_num)).to(0)
    #     self.i1s_r = self.i1s_r * (1 - mask)

    #     self.i1s_yhat = torch.exp(self.i1s_delta).unsqueeze(0) + torch.sum(i1c_mu * torch.exp(self.i1s_r), dim=-1)

    #     # i1s_r_sum = torch.exp(self.i1s_r)
    #     # mask = torch.diag(torch.ones(cls_num)).to(0)
    #     # i1s_r_sum = i1s_r_sum * (1 - mask)
       
    #     # self.i1s_r_sum = torch.sum(i1s_r_sum, dim=-1)
    #     # print ('original : ', self.i1s_yhat.norm())
    #     pred_recover = 1. / self.i1s_yhat
    #     print ('recover : ', pred_recover[0])
    #     return pred_recover + self.q.unsqueeze(0) * self.gamma[1], None

    def interaction_1_s2c(self, recv):
        # Addtional Round 2: Server -> Client

        i1c_mu, alpha = recv
        batch_size, cls_num, _ = i1c_mu.size()

        self.i1s_delta = torch.zeros(cls_num).to(0) 
        diff = self.rcls.unsqueeze(0).repeat(cls_num, 1) - self.rcls.unsqueeze(-1)
        
        self.i1s_r = self.i1s_delta.unsqueeze(0).unsqueeze(-1) - diff.unsqueeze(0) * alpha.unsqueeze(-1).unsqueeze(-1)
        
        mask = torch.diag(torch.ones(cls_num)).to(0)
        self.i1s_r = self.i1s_r * (1 - mask)

        self.i1s_yhat = torch.exp(self.i1s_delta).unsqueeze(0) + torch.sum(i1c_mu * torch.exp(self.i1s_r), dim=-1)

        i1s_r_sum = torch.exp(self.i1s_r)
        mask = torch.diag(torch.ones(cls_num)).to(0)
        i1s_r_sum = i1s_r_sum * (1 - mask)
       
        self.i1s_r_sum = torch.sum(i1s_r_sum, dim=-1)
        # print ('original : ', self.i1s_yhat.norm())
        pred_recover = 1. / self.i1s_yhat
        return pred_recover + self.q.unsqueeze(0) * self.gamma[1], self.i1s_r_sum


    
    def post(self, y_gt, diff):
        i1c_mu, alpha = self.interaction_1_c2s()
        i1s_yhat, i1s_r_sum = self.interaction_1_s2c((i1c_mu, alpha))
        # print ('i1s_yhat : ', i1s_yhat)
        grad_L2ylast = i1s_yhat - y_gt
        # print ('right ? ', i1c_mu.norm(), alpha.norm(), self.rcls.norm(),torch.einsum('k,c->kc', alpha, self.rcls).norm(), grad_L2ylast.norm())
        self.post_temp(grad_L2ylast)

        if diff:
            # print ('diff of what')
            for m in self.named_modules():
                # if type(m[1]) is Linear:
                #     m[1].fc.weight.grad = (m[1].rectify_grad - gamma[0] * m[1].sigma_conv_a - gamma[1] * m[1].beta + gamma[1] * gamma[0] * m[1].sigma_conv_b) * m[1].r
                # elif type(m[1]) is Conv2d:
                #     m[1].conv.weight.grad = (m[1].rectify_grad - gamma[0] * m[1].sigma_conv_a - gamma[1] * m[1].beta + gamma[1] * gamma[0] * m[1].sigma_conv_b) * m[1].r.unsqueeze(-1).unsqueeze(-1)
                if type(m[1]) is Linear or type(m[1]) is Conv2d:
                    if type(m[1]) is Linear:
                        r_max, r_min = m[1].r.max(), m[1].r.min()
                        # r_max, r_min = 2., 1.
                    else:
                        r_max, r_min = m[1].r.max(), m[1].r.min()
                        # r_max, r_min = 2., 1.

                    appro_norm = r_max * (m[1].rectify_grad.abs().max() + m[1].sigma_conv_a.abs().max() + m[1].sigma_conv_b.abs().max() + m[1].beta.abs().max())
                    # appro_norm = 2.
                    m[1].rectify_grad, m[1].sigma_conv_a, m[1].sigma_conv_b, m[1].beta = \
                        m[1].rectify_grad / appro_norm, m[1].sigma_conv_a / appro_norm, m[1].sigma_conv_b / appro_norm, m[1].beta / appro_norm
                    
                    # if type(m[1]) is Linear:
                    #     # sigma_d, sigma_eta = 1., 1.
                    #     sigma_d, sigma_eta = 1. / r_min, 1. / r_min 
                    # else:
                    #     # sigma_d, sigma_eta = 1., 1. 
                    #     sigma_d, sigma_eta = 1. / r_min, 1. / r_min

                    if type(m[1]) is Conv2d:
                        # noise = torch.distributions.normal.Normal(0., 5e-1 * sigma_eta).sample() 
                        noise = rand(1, 3, False).to(0) * 5e-1
                    else:
                        noise = rand(1, 3, False).to(0) * 0
                        # noise = torch.distributions.normal.Normal(0., 5e-1 * sigma_eta).sample() 

                    m[1].rectify_grad += noise

                    m[1].post_data = (m[1].sigma_conv_a, m[1].sigma_conv_b, m[1].beta)

    def fl_modules(self):
        module_dict = {}
        for idx, m in enumerate(self.named_modules()):
            if type(m[1]) is Conv2d or type(m[1]) is Linear:
                # print (idx, '->', m)
                # module_list.append(m)
                module_dict[m[0]] = m[1]
        return module_dict




def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
