import os
import importlib
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


def get_mnist(batch_size):
    train_dataset = datasets.MNIST(root='../../data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True
                                   )

    test_dataset = datasets.MNIST(root='../../data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True
                                  )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True
                              )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False
                             )

    return train_loader, test_loader


def get_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='',
                                                 train=True,
                                                 transform=transform,
                                                 download=True)

    test_dataset = torchvision.datasets.CIFAR10(root='',
                                                train=False,
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


# Functions for loading DICOM files:
def get_dicom_files(path, string1, string2):
    """
    Returns a list of filenames for all DICOM files in 'path/'.
    Filenames containing specific strings can be selected via string1, string2
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if string1 in filename:
                if string2 in filename:
                    file_list.append(os.path.join(root, filename))
    return file_list


def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    accuracy = []

    for batch, (images, labels) in enumerate(dataloader):
        # Forward pass:
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backpropagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running training accuracy for visualisation:
        #predictions = pred.argmax(1)
        #num_correct = (predictions == labels).sum()
        #running_training_acc = float(num_correct) / float(images.shape[0])
        #accuracy.append(running_training_acc)

        # Some metrics:
        #num_iter = epoch * len(dataloader) + batch
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(images)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # Tensorboard:
            #tb.add_scalar("Training loss per 100 batches", loss, global_step = num_iter)
            #tb.add_scalar("Training accuracy per 100 batches", running_training_acc, global_step = num_iter)


def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Average loss: {test_loss:>8f} \n")


def init_obj_cls(string_def):
    obj_cls = getattr(importlib.import_module(string_def))
    return obj_cls


def init_obj(string_def, params):
    obj_cls = init_obj_cls(string_def)
    if params is None:
        params = {}
    return obj_cls(**params)
