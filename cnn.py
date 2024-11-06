import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torchvision.datasets as dset
import numpy as np

class Model(nn.Module):
    def __init__(self, num_passes=100, max_iter=500, batch_size=20, report_interval=10):
        super(Model, self).__init__()
        
        # 设置训练参数
        self.num_passes = num_passes
        self.max_iter = max_iter 
        self.batch_size = batch_size
        self.report_interval = report_interval
        
        # Convolution + average pooling
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
     
        # Convolution + max pooling
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1600, 100)
        self.fc2 = nn.Linear(100, 10)

        # 初始化优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        
        # 初始化日志记录
        self.train_log_loss = {}
        self.test_log_loss = {}
        self.train_log_acc = {}
        self.test_log_acc = {} 
    
    def forward(self, x):
        # Convolution + average pooling
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.avgpool1(out)
        
        # Convolution + max pooling
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        # resize
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        
        # full connect layers
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out
    
    def yield_batches(self, features, classes, batchsize):
        sets = np.arange(features.shape[0])
        np.random.shuffle(sets)
        for i in range(0, features.shape[0] - batchsize + 1, batchsize):
            e = sets[i:i + batchsize]
            yield torch.FloatTensor(features[e]), torch.LongTensor(classes[e])

    def run_model(self):
        n_iter = 0
        for i in range(1, self.num_passes + 1):
            for x, y in self.yield_batches(self.X_train, self.y_train, self.batch_size):
                n_iter += 1
                self.forward(x)
                self.backward(y)
                if n_iter % self.report_interval == 0:
                    self.forward(self.X_train_sub)
                    yield n_iter, [layer.vals for layer in self.hidden_layers]

            self.forward(self.X_test)
            self.test_log_loss[i] = self.loss(self.y_test)
            self.test_log_acc[i] = self.accuracy(self.y_test)

            self.forward(self.X_train)
            self.train_log_loss[i] = self.loss(self.y_train)
            self.train_log_acc[i] = self.accuracy(self.y_train)
            print("Epoch: {}, Train Acc: {}, Test Acc: {}".format(i, self.train_log_acc[i], self.test_log_acc[i]))

            if n_iter > self.max_iter:
                break