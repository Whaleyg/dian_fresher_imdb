# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_train
from vocab import Vocab
from torch.utils.tensorboard import SummaryWriter

global j
j = 0
voc_model = pickle.load(open("./models/vocab1.pkl", "rb"))
sequence_max_len = 100
Vocab()
writer = SummaryWriter('./log')
epochs = 50
learning_rate = 2
train_batch_size = 128
test_batch_size = 64
loss_fn = nn.CrossEntropyLoss()


def collate_fn(batch):
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataset():
    return dataset_train.ImdbDataset(train)


def get_dataloader(train=True):
    imdb_dataset = get_dataset()
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings=len(voc_model), embedding_dim=300, sparse=True)
        self.fc = nn.Linear(300, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        out = self.fc(embedded)
        out = self.dropout(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImdbModel().to(device)


def train(imdb_model, epoch):
    global j
    train_dataloader = get_dataloader(train=True)
    optimizer = torch.optim.SGD(imdb_model.parameters(), lr=learning_rate)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (data, target) in enumerate(bar):
        j += 1
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = imdb_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        writer.add_scalar('train_loss', loss, j)
        optimizer.step()
        bar.set_description("epoch:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))


def tst(imdb_model, epoch):
    test_loss = 0
    correct = 0
    imdb_model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device)
            target = target.to(device)
            output = imdb_model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        accuracy = float(correct) / len(test_dataloader.dataset)
        if (i + 1) % 2 == 0:
            writer.add_scalar('accuracy', accuracy, k)
        test_loss /= len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    imdb_model = ImdbModel().to(device)
    for i in range(epochs):
        train(imdb_model, epochs)
        if (i + 1) % 2 == 0:
            k = (i + 1) / 2
            tst(imdb_model, epochs)
