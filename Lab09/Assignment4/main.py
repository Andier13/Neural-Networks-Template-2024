import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets  # for Mist
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # To inherit our neural network
from torch.utils.data import DataLoader  # For management of the dataset (batches)
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, RandomAffine
from tqdm import tqdm  # For nice progress bar!

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 500
dropout_rate = 0.1
regularization_rate = 0.0001


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        self.sequence = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(250, 125),
            nn.BatchNorm1d(125),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(125, num_classes)
        )

    def forward(self, x):
        return self.sequence.forward(x)


# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
#
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device:{torch.cuda.current_device()}")
#
# print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
#

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

test_transform = lambda x: torch.from_numpy(np.array(x, dtype=np.float32).flatten() / 255)
train_transform = Compose([
    RandomAffine(degrees=10, translate=(0.1, 0.1)),
    test_transform
])

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=train_transform, download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=test_transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=regularization_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)
            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            # Check how many we got correct
            num_correct += (predictions == y).sum()
            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

def save_predictions(loader, model):
    model.eval()

    pred_list = []
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)
            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)
            pred_list.append(predictions)

    predictions1 = torch.concat(pred_list)
    print(predictions1)

    data = {
        "ID": [],
        "target": [],
    }
    # num_correct = 0
    # num_samples = 0
    for i, (image, label) in enumerate(test_dataset):
        data["ID"].append(i)
        data["target"].append(predictions1[i].item())
        # num_correct += (label == predictions1[i])
        # num_samples += 1

    # print((num_correct / num_samples).item())

    df = pd.DataFrame(data)
    df.to_csv("submission.csv", index=False)
    model.train()


max_accuracy = 0

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # Get to correct shape
        data = data.reshape(data.shape[0], -1)
        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent or adam step
        optimizer.step()

    test_accuracy = check_accuracy(test_loader, model)
    print(f"epoch = {epoch}; test = {test_accuracy}")

    if test_accuracy > max_accuracy:
        max_accuracy = test_accuracy
        save_predictions(test_loader, model)
