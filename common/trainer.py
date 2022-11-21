# coding: utf-8
import sys, os

sys.path.append(os.pardir)
from common.util import Timer, Accumulator, init_weights, try_gpu, correct, evaluate_accuracy_gpu
from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchmetrics import Accuracy


class Trainer:
    def __init__(self, net, train_iter, test_iter,
                 num_epochs=50, batch_size=128,
                 optimizer=torch.optim.SGD, lr = 0.01,weight_decay = 0,lr_scheduler_enabled = False, device = try_gpu(),
                 verbose=True):
        self.net = net
        self.net.apply(init_weights)
        self.device = device
        print('training on', self.device)
        net.to(self.device)
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr_scheduler_enabled = lr_scheduler_enabled
        self.verbose = verbose
        self.timer = Timer()
        # self.accuracy = Accuracy().to(device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.net.parameters(), lr=lr,weight_decay= weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer,T_max=10)

        self.train_l = 0
        self.train_acc = 0
        self.test_acc = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.metric = Accumulator(3)

    def train_step(self):
        for (X, y) in self.train_iter:
            self.timer.start()
            self.optimizer.zero_grad()
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            l.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.metric.add(l * X.shape[0], correct(y_hat, y), X.shape[0])

            self.timer.stop()


    def train(self):
        for epoch in range(self.num_epochs):
            self.train_step()
            if self.lr_scheduler_enabled:
                self.scheduler.step()
            self.train_l = self.metric[0] / self.metric[2]
            self.train_acc = self.metric[1] / self.metric[2]
            self.metric.reset()
            self.train_acc_list.append(self.train_acc)
            self.train_loss_list.append(self.train_l)
            self.test_acc = evaluate_accuracy_gpu(self.net, self.test_iter)
            self.test_acc_list.append(self.test_acc)
            if self.verbose:
                print("第{}个epoch的学习率：{}, train_loss:{}, train_acc:{}, test_acc: {}"
                  .format(epoch + 1, self.optimizer.param_groups[0]['lr'],self.train_l, self.train_acc, self.test_acc))


        print("=============== Final Test Accuracy ===============")
        print("test acc:" + str(self.test_acc))
        print('{} examples/sec on {}'.format(len(self.train_iter.dataset) * self.num_epochs/ self.timer.sum(), str(self.device)))

