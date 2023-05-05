import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 加载已经训练好的 PyTorch 模型
model = torch.load('path/to/model.th')
model.eval()

# 加载攻击后的测试集数据
test_data = ImageFolder('path/to/attacked_images', transform=transforms.ToTensor())

# 初始化计数器
num_correct = 0
total = 0

# 遍历攻击后的测试集图像，计算 ASR
for images, labels in test_data:
    # 将攻击后的图像输入到模型中进行分类
    with torch.no_grad():
        outputs = model(images)

    # 检查模型的分类结果是否匹配真实标签
    _, predicted = torch.max(outputs.data, 1)
    num_correct += (predicted == labels).sum().item()
    total += 1

# 计算攻击成功率
asr = num_correct / total
print('攻击成功率为:', asr)
