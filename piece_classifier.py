import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torchvision import transforms
import torch.nn.functional as F

device = 'cpu'

class PieceClassifier(nn.Module):
    """Binary classifier to determine whether a square has a piece on it or not"""
    def __init__(self):
        super(PieceClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_piece_classifier(images, labels):
    model = PieceClassifier().to(device)

    images = torch.Tensor(images)
    labels = torch.Tensor(labels)
    dataset = TensorDataset(images, labels) # create datset
    dataloader = DataLoader(dataset, batch_size=512)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n = len(dataset)

    for epoch in range(1000):  # loop over the dataset multiple times
        #TODO break once we reach 100% accuracy
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #import pdb;pdb.set_trace()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs[:, 0] > 0
            l = labels[:, 0] == 1
            correct = torch.sum(pred == l).item()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f acc: %3f' %
            (epoch + 1, i + 1, running_loss, correct/n))
        if correct == n:
            print("Training complete")
            break
        running_loss = 0.0
    return model

def get_piece_prediction(model, image):
    image = torch.Tensor(image)
    return model(image)[0][0].item()



    
