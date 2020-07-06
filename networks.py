import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from focal_loss import FocalLoss
from apex import amp

def train_var_2(name:str, model:torch.nn.Module, dataset:torch.utils.data.Dataset,
                 criterion, optimizer, scheduler,
                 epochs:int, batch_size:int, epoch_size:int):

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                         pin_memory=False, drop_last=True)

    for epoch in range(epochs):

        running_loss = []

        for idx, (inputs, labels) in enumerate(loader, 0):

            inputs, labels = inputs.cuda(), labels.long().cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            running_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

            if idx > epoch_size:
                break

        mean_loss = np.mean(running_loss) / 2
        print(f'Epoch {epoch} - loss {mean_loss}')

    print(f'{name} finished training')



def test(model :torch.nn.Module, dataset: torch.utils.data.Dataset, device:torch.device, path:str, size:int):

    criterion = torch.nn.CrossEntropyLoss()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, replacement=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

    running_loss = 0.0
    total = 0

    rows = []

    with torch.no_grad():

        for idx, data in enumerate(loader, 0):
            # Get the inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            print(outputs.shape)
            max, index = outputs.max(1)
            loss = criterion(outputs, labels)
            real_score = outputs[0, 0].item()
            fake_score = outputs[0, 1].item()
            fake_prob = np.exp(fake_score) / (np.exp(fake_score) + np.exp(real_score))
            row =(loss.item(), real_score, fake_score, labels.item(), index.item(), fake_prob)
            rows.append(row)
            # print statistics
            running_loss+=loss
            if idx > size:
                total = idx
                break

        # Print results at the end of the epoch
        print('Finished Validation of Model.')
        print('Total execution', running_loss / (idx+1))

    res = pd.DataFrame(rows, columns = ('loss', 'score:original', 'score_fake', 'labels', 'predicted', 'fake_prob'))
    res.to_csv(path, index = False)