import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from sklearn.cluster import KMeans

def calculate_feat_dist(x_r: torch.Tensor, mu_c: torch.Tensor) -> float:
    """
    计算重建图像与目标类别中心向量之间的特征距离

    Parameters
    ----------
    x_r : torch.Tensor
        重建图像的像素张量，shape 为 (C, H, W)
    mu_c : torch.Tensor
        目标类别的中心向量，shape 为 (C, H, W)

    Returns
    -------
    float
        特征距离
    """
    feat_dist = ((x_r - mu_c) ** 2).sum()  # 计算像素值差的平方和
    return feat_dist.item()

def calculate_recon_error(recon_image: Image, target_class: int) -> float:
    """
    计算重构图像与目标类别中心向量之间的特征距离

    Parameters
    ----------
    rrecon_image : Image
        重构图像的像素图像
    target_class : int
        目标类别的编号

    Returns
    -------
    float
        特征距离
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    recon_image = transform(recon_image)

    # 读取 CIFAR-10 数据集，并计算每个类别的中心向量
    dataset = datasets.CIFAR10(root='~/.torch', train=True, download=True, transform=None)
    images = torch.stack([transforms.ToTensor()(img) for img, _ in dataset])
    labels = torch.tensor([label for _, label in dataset])

    num_classes = 10
    centroids_file = 'centroids.pt'

    if not os.path.exists(centroids_file):
        print('开始计算并保存中心向量...')
        centroids = []
        for i in range(num_classes):
            idx = labels == i
            images_i = images[idx]

            kmeans = KMeans(n_clusters=1)
            kmeans.fit(images_i.reshape(images_i.shape[0], -1))

            centroid = torch.from_numpy(kmeans.cluster_centers_.reshape(images.shape[1:]))
            centroids.append(centroid)

        centroids = torch.stack(centroids)
        torch.save(centroids, centroids_file)
    else:
        print('从文件中读取中心向量...')
        centroids = torch.load(centroids_file)

    mu_c = centroids[target_class]

    feat_dist = calculate_feat_dist(recon_image, mu_c)
    return feat_dist
