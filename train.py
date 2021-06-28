import torch
from torchvision import models, transforms
from torchmetrics import F1

from ut.config import *
from ut.dataset import SportDataset
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import os
import time
import copy
from tqdm import tqdm
from datetime import datetime

def get_dataloader(path_to_train_csv, path_to_test_csv, batch_size=1):
    print('Getting Dataloader')

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = SportDataset(path_to_train_csv, transform = train_transform)
    test_dataset = SportDataset(path_to_test_csv, transform = test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    dataloaders = {'train':train_dataloader, 'val':test_dataloader}
    return dataloaders


def train(dataloaders):
    print('Train')

    # creating dir for weights
    today = datetime.now()
    save_dir = 'resnet50_'+today.strftime("%Y-%m-%d-%H:%M:%S")
    os.makedirs(save_dir, exist_ok=True)

    since = time.time()
    if is_cuda:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # getting resnet50
    net = models.resnet50(pretrained=True)
    num_ftrs = net.fc.in_features
    # changing last layer
    net.fc = torch.nn.Linear(num_ftrs, num_classes)
    net.to(device)
    

    best_model_wts = copy.deepcopy(net.state_dict())
    best_f1 = 0.0

    # F1 macro score
    f1 = F1(num_classes=num_classes, average='macro')
    f1.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train/val loop
    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0 
            running_f1 = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                f1(preds, labels.data)

            # calculating epoch values
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1.compute()

            print('{} Loss: {:.4f} Acc: {:.5f} F1: {:.5f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1))

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(net.state_dict())

        print()
        #saving each epoch
        torch.save(net, os.path.join(save_dir, 'epoch_%d.pt'%epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # load best model weights and saving model
    net.load_state_dict(best_model_wts)
    torch.save(net, os.path.join(save_dir, 'best_model.pt'))

    return net



    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_train_csv', default=None, type=str)
    parser.add_argument('--path_to_test_csv', default=None, type=str)
    parser.add_argument('--batch_size', default=None, type=int)

    args = parser.parse_args()

    dataloaders = get_dataloader(args.path_to_train_csv, args.path_to_test_csv, args.batch_size)

    best_model = train(dataloaders)
    torch.save(best_model, 'best_model.pt')