from torch.utils.tensorboard import SummaryWriter
from data import make_dataloader
from model import ESIM
import torch
import torch.nn as nn

learning_rate = 1e-4
embedding_dim = 50
len_hidden = 50
batch_size = 64
epoch = 40

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

train_data, test_data, glove = make_dataloader(batch_size=batch_size)

model = ESIM(embedding_dim=embedding_dim, len_hidden=len_hidden, len_words=glove.len_words, longest=glove.longest,
             weight=torch.tensor(glove.embedding, dtype=torch.float))
model = model.to(device)

loss_func = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_train_step = 0

writter = SummaryWriter('logs')

for i in range(epoch):

    print('------------第{}轮训练开始'.format(i + 1))
    model.train()

    for data in train_data:
        x1, x2, y = data
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        pred = model(x1, x2)

        optimizer.zero_grad()
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 500 == 0:
            print('训练次数: {} , loss: {} '.format(total_train_step, loss.item()))
            writter.add_scalar('train_loss', loss.item(), total_train_step)

    model.eval()
    test_loss = 0
    test_acc = 0
    test_step = 0
    with torch.no_grad():
        for data in test_data:
            x1, x2, y = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            pred = model(x1, x2)
            loss = loss_func(pred, y)
            test_loss = test_loss + loss.item()
            test_step = test_step + 1
            acc = (pred.argmax(1) == y).sum()
            test_acc = test_acc + acc

        test_loss = test_loss / test_step
        test_acc = test_acc / len(test_data.dataset)
        print('测试:  loss:{} ,正确率: {} '.format(test_loss, test_acc))
        writter.add_scalar('test_loss', test_loss, i + 1)
        writter.add_scalar('test_acc', test_acc, i + 1)

writter.close()
