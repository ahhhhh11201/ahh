import torch
from torch import nn
from torchvision import models


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


class BPNNModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel, self).__init__()

        self.linear1 = nn.Linear(28 * 28, 128)
        # 在第一个隐层使用ReLU激活函数
        self.relu1 = nn.ReLU()
        """
                定义第二个线性层，
                输入是第一个隐层的输出，
                输出为第二个隐层的输入，大小为64。
                """
        self.linear2 = nn.Linear(128, 64)
        # 在第二个隐层使用ReLU激活函数
        self.relu2 = nn.ReLU()
        """
        定义第三个线性层，
        输入是第二个隐层的输出，
        输出为输出层，大小为10
        """
        self.linear3 = nn.Linear(64, 10)
        # 最终的输出经过softmax进行归一化
        self.softmax = nn.LogSoftmax(dim=1)
        self.model = nn.Sequential(nn.Linear(28 * 28, 128),
                               nn.ReLU(),
                               nn.Linear(128, 64),
                               nn.ReLU(),
                               nn.Linear(64, 10),
                               nn.LogSoftmax(dim=1)
                               )

    def forward(self, x):
        """
        定义神经网络的前向传播
        x: 图片数据, shape为(64, 1, 28, 28)
        """
        # 首先将x的shape转为(64, 784)
        x = x.view(x.shape[0], -1)
        # 接下来进行前向传播
        x = self.model(x)
        return x

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.Sequential(nn.LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True),
                                       nn.Dropout(0.2),

                                       )




class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self, ):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons,device="cuda"))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()

        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
        out = self.FC(self.hidden)

        return out.view(-1, self.n_outputs)  # batch_size X n_output


def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        # model = models.resnet18(pretrained=pretrained)
        model = Model()
    elif name == "Rnn":
        # model = models.resnet50(pretrained=pretrained)
        model = ImageRNN(32,28,28,150,10)
    elif name == "bp":
        model = BPNNModel()
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model