# resnet

ResNet-18 adn ResNet-50 model from
"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

For the Pytorch implementation, you can refer to [pytorchx/resnet](https://github.com/wang-xinyu/pytorchx/tree/master/resnet)

Following tricks are used in this resnet, nothing special, residual connection and batchnorm are used.

- Batchnorm layer, implemented by scale layer.

```
1.重新训练生成pth文件
直接更改 train文件夹里的数据，
更改train.py的model.fc = nn.Linear(512, 5)这行代码的5，改为你要输出的种类，点击train.py即可训练。

2.通过运行generate_wts.py即可通过训练到的pth生成wts文件。

3.然后通过wts文件生成engine文件：
先更改static const int OUTPUT_SIZE = 5;的5
再更改IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 5, weightMap["fc.weight"], weightMap["fc.bias"]);的5
如果更改了输入图片的尺寸，要先更改全局平均池化层大小，
通过train.py的模型结构输出
             ReLU-65            [-1, 512, 4, 2]               0
       BasicBlock-66            [-1, 512, 4, 2]               0
AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
           Linear-68                    [-1, 5]           2,565
可以看到当前模型全局平均池化层的大小，比如这里的就是4*2
然后把resnet18.cpp里的IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{4, 2});
(4,2)改成对应的大小

4.接着编译生成engine文件

mkdir build

cd build

cmake ..

make

sudo ./resnet18 -s   // serialize model to plan file i.e. 'resnet18.engine'
sudo ./resnet18 -d   




