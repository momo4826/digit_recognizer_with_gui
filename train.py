import torch.optim

from dataset.mnist import gen_loader
import matplotlib.pyplot as plt
from models import MLPNet, LeNet
from common.trainer import Trainer

# constants
num_epochs = 10
batch_size = 256
lr = 1e-3
weight_decay = 0
optimizer=torch.optim.Adam
train_iter = gen_loader(batch_size, 'train')
test_iter = gen_loader(batch_size, 'test')
model = LeNet()

if __name__ == '__main__':
    trainer = Trainer(model, train_iter, test_iter,num_epochs,batch_size,optimizer,lr,weight_decay)
    trainer.train()
    torch.save(model.state_dict(), 'MLPNet_weights.pth')
    print("Saved Network Parameters!")

    # 绘制图形
    x = list(range(num_epochs))
    plt.plot(x, trainer.train_acc_list, color = 'blue', marker='o', label='train_acc', markevery=2)
    plt.plot(x, trainer.test_acc_list, color = 'red', marker='o', label='test_acc', markevery=2)
    plt.plot(x, trainer.train_loss_list, color = 'green', marker='+', label='train_loss', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
