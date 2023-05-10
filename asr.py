import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensor
import resnet_v2 as resnet
from PIL import Image
import torchvision.datasets as datasets
from dataset import CIFAR10Test,CIFAR10DatasetTrain

# 加载已经训练好的 PyTorch 模型
model = resnet.__dict__["resnet20"]().cuda()
checkpoint = torch.load('path/to/model.th')

# print('checkpoint',len(checkpoint['state_dict']))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# img_path = 'path/to/attacked_images/9/min_loss_1683291805.png'
# img = Image.open(img_path)
# print(img)
# img.show()

transform_test = Compose([
        Resize(64, 64),
        ToTensor()
    ])
# 加载攻击后的测试集数据
# test_data = CIFAR10Test('path/to/df_attacked_images', transform=transform_test)
test_data = datasets.CIFAR10(root='~/.torch', train=False,transform=transform_test)

test_data = CIFAR10DatasetTrain(dataset=test_data,transform=transform_test,test=True)

val_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# 初始化计数器
num_correct = 0
total = 0

# 遍历攻击后的测试集图像，计算 ASR
for images, target in val_loader:
    images = images.to('cuda', dtype=torch.float)
    # images = images.unsqueeze(0).to('cuda')  
    target = target.to('cuda')
    # print('target shape: ',target.shape)

    # print('image:',images,'label:',labels)
    # 将攻击后的图像输入到模型中进行分类
    with torch.no_grad():
        outputs = model(images)

    # 检查模型的分类结果是否匹配真实标签
    # print('outputs',outputs.data.shape)
    # print('labels',labels.data.shape)
    predicted = torch.argmax(outputs.data, 1)
    # labels = torch.argmax(labels.data, 1)
    # print('outputs',predicted.shape)

    
    num_correct += (predicted == target).sum().item()
    total += 1
    print("predicted: ",predicted,"labels: ",target)
    if total >100:
        break

# 计算攻击成功率
asr = num_correct / total
print('攻击成功率为:', asr)
