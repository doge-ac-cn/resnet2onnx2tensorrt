import torch
import torchvision
import torchvision.models as models
from torch import nn
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import trange




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5)
    model = model.to(device)
    # summary 
    from torchsummary import summary
    summary(model, (3,128, 64)) 
    model.train(mode=True)
    # print(model)
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder('train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=True )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    # 迭代epoch
    acc_list = []
    for epoch in trange(10):

        correct = 0
        total = 0
        correct_val = 0
        total_val = 0
        model.train()
        for i, (img, label), in (enumerate(dataloader, 0)):
            # get the input
            inputs = img.to(device)
            labels = label.to(device)

            # zeros the paramster gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # 计算loss
            loss.backward()  # loss 求导
            optimizer.step()  # 更新参数
            scheduler.step()
            # print statistics

            prediction = torch.argmax(outputs, 1)
            correct += (prediction == labels).sum().float()
            total += len(labels)
        print('Train Accuracy: %f' % ((correct / total).cpu().detach().data.numpy()))

# 保存
torch.save(model, 'resnet18.pth')



# 转onnx
dummy_input = torch.randn(1, 3, 128, 64, device='cuda')
torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=["imgs"],
                  output_names=["preds"])
