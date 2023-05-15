import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
# from sklearn.cluster import KMeans
import resnet_v2 as resnet
# from dataset import CIFAR10DatasetTrain
from torch.utils.data import DataLoader
# def calculate_feat_dist(x_r: torch.Tensor, mu_c: torch.Tensor) -> float:
#     """
#     计算重建图像与目标类别中心向量之间的特征距离

#     Parameters
#     ----------
#     x_r : torch.Tensor
#         重建图像的像素张量, shape 为 (C, H, W)
#     mu_c : torch.Tensor
#         目标类别的中心向量, shape 为 (C, H, W)

#     Returns
#     -------
#     float
#         特征距离
#     """
#     feat_dist = ((x_r - mu_c) ** 2).sum()  # 计算像素值差的平方和
#     return feat_dist.item()

# def calculate_recon_error(recon_image: Image, target_class: int) -> float:
#     """
#     计算重构图像与目标类别中心向量之间的特征距离

#     Parameters
#     ----------
#     rrecon_image : Image
#         重构图像的像素图像
#     target_class : int
#         目标类别的编号

#     Returns
#     -------
#     float
#         特征距离
#     """

#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor()
#     ])
#     recon_image = transform(recon_image)

#     # 读取 CIFAR-10 数据集，并计算每个类别的中心向量
#     dataset = datasets.CIFAR10(root='~/.torch', train=True, download=True, transform=None)
#     images = torch.stack([transforms.ToTensor()(img) for img, _ in dataset])
#     labels = torch.tensor([label for _, label in dataset])

#     num_classes = 10
#     centroids_file = 'centroids.pt'

#     if not os.path.exists(centroids_file):
#         print('开始计算并保存中心向量...')
#         centroids = []
#         for i in range(num_classes):
#             idx = labels == i
#             images_i = images[idx]

#             kmeans = KMeans(n_clusters=1)
#             kmeans.fit(images_i.reshape(images_i.shape[0], -1))

#             centroid = torch.from_numpy(kmeans.cluster_centers_.reshape(images.shape[1:]))
#             centroids.append(centroid)

#         centroids = torch.stack(centroids)
#         torch.save(centroids, centroids_file)
#     else:
#         print('从文件中读取中心向量...')
#         centroids = torch.load(centroids_file)

#     mu_c = centroids[target_class]

#     feat_dist = calculate_feat_dist(recon_image, mu_c)
#     return feat_dist


def get_feat_dist(model,target):

    # 当前模型的y_last向量和logits向量
    y_last = model.y_last
    logits = model.logits

    # feat_mean保存路径
    feat_mean_path = 'feat_mean.pt'
    # 如果已经计算过特征均值，则直接从文件中读取
    if os.path.exists(feat_mean_path):
        feat_mean = torch.load(feat_mean_path)
        y_last_mean = torch.from_numpy(feat_mean['y_last_mean']).cuda()
        logits_mean = torch.from_numpy(feat_mean['logits_mean']).cuda()

    # 否则，计算特征均值并保存到文件中
    else:
        y_last_mean, logits_mean = calculate_feat_mean()


    for i in range(y_last.shape[0]):
        y_last_dist = ((y_last[i] - y_last_mean[target[i]]) ** 2).sum()
        logits_dist = ((logits[i] - logits_mean[target[i]]) ** 2).sum()
        print('y_last_dist:', y_last_dist.item())
        print('logits_dist:', logits_dist.item())

    # 计算特征距离
    return y_last_dist,logits_dist


def get_modle():
    # 加载已经训练好的 PyTorch 模型
    model = resnet.__dict__["resnet20"]().cuda()
    checkpoint = torch.load('path/to20_seed42/model.th')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def calculate_feat_mean():
    model = get_modle()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    # 读取 CIFAR-10 数据集，并通过模型得到每个类别的y_last和logits
    dataset = datasets.CIFAR10(root='~/.torch', train=False, download=True, transform=transform)
    # test_data = CIFAR10DatasetTrain(dataset=dataset,transform=transform,test=True)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # 初始化类别的logits向量总和、y_last向量总和样本数量
    logits_sum = torch.zeros((10, 10)).cuda()
    y_last_sum = torch.zeros((10, 64)).cuda()
    # feat_sum = torch.zeros((10, 64)).cuda()
    feat_count = torch.zeros(10).cuda()

    # 循环遍历数据集，计算每个类别的输出向量均值
    with torch.no_grad():
        for images, labels in data_loader:
            # print('images:',images.shape,'labels:',labels.shape)
            images = images.cuda()
            outputs = model(images)

            logits = model.logits
            y_last = model.y_last
            for i in range(len(outputs)):
                logits_sum[labels[i]] += logits[i]
                y_last_sum[labels[i]] += y_last[i]
                feat_count[labels[i]] += 1

    # 计算每个类别的logits向量和y_last向量均值
    logits_mean = logits_sum / feat_count.view(-1, 1)
    y_last_mean = y_last_sum / feat_count.view(-1, 1)

    # 保存结果，用字典存储
    result = {
        'logits_mean': logits_mean.cpu().numpy(),
        'y_last_mean': y_last_mean.cpu().numpy()
    }
    torch.save(result, 'feat_mean.pt')

    print('y_last_mean', y_last_mean)
    print('logits_mean', logits_mean)
    return y_last_mean,logits_mean


def get_feat_mean_by_label(label):
    feat_mean = torch.load('feat_mean.pt')
    logits_mean = feat_mean['logits_mean']
    y_last_mean = feat_mean['y_last_mean']
    return logits_mean[label],y_last_mean[label]

def get_feat_mean():
    feat_mean = torch.load('feat_mean.pt')
    logits_mean = feat_mean['logits_mean']
    y_last_mean = feat_mean['y_last_mean']
    return y_last_mean,logits_mean

def get_y_last_mean():
    feat_mean = torch.load('feat_mean.pt')
    y_last_mean = feat_mean['y_last_mean']
    return y_last_mean

def get_logits_mean():
    feat_mean = torch.load('feat_mean.pt')
    logits_mean = feat_mean['logits_mean']
    return logits_mean


if __name__ == '__main__':
    y_last_mean,logits_mean = calculate_feat_mean()
    


