import torch
import pandas as pd
import numpy as np

def train_var_2(name:str, model:torch.nn.Module, dataset:torch.utils.data.Dataset,
                   epochs:int, batch_size:int, epoch_size:int, power:int, device:torch.device):

    # TODO: Label smoothing
    # TODO: Focal loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda x: (1-x/epochs)**power, last_epoch=-1)

    sampler = torch.utils.data.RandomSampler(dataset, replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    for epoch in range(epochs):

        running_loss = []

        for idx, (inputs, labels) in enumerate(loader, 0):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # print statistics
            running_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

            if idx > epoch_size:
                break

        mean_loss = np.mean(running_loss)
        print(f'Epoch {epoch} - loss {mean_loss}')

    print(f'{name} finished training')

def train_var_1(title:str, model :torch.nn.Module, dataset: torch.utils.data.Dataset, epochs: int, batch:int,  device:torch.device):
    # Cross Entropy Loss plays the same role as Softmax loss (multiclass regression)
    # With this we got two classes: {FAKE, REAL}. An the algorithm should spit the probablities.
    criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='mean').to(device)
    # optim.SGD(net.parameters(), lr=.0002, amsgrad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002, amsgrad=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch)

    for epoch in range(epochs):

        running_loss = []

        for idx, data in enumerate(loader, 0):
            # Get the inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # print statistics
            running_loss.append(loss.item())
            optimizer.step()

        mean_loss = np.mean(running_loss)
        print(f'Epoch {epoch} - loss {mean_loss}')

    print(f'{title} finished training')


def test(model :torch.nn.Module, dataset: torch.utils.data.Dataset, df:pd.DataFrame, device:torch.device, path:str):

    criterion = torch.nn.CrossEntropyLoss(weight=None).to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    running_loss = .0

    videos = df.video.tolist()
    paths = df.path.tolist()

    rows = []

    with torch.no_grad():

        for idx, data in enumerate(loader, 0):
            # Get the inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            # forward + backward + optimize
            outputs = model(inputs)
            max, index = outputs.max(1)
            loss = criterion(outputs, labels)
            real_score = outputs[0, 0].item()
            fake_score = outputs[0, 1].item()
            fake_prob = np.exp(fake_score) / (np.exp(fake_score) + np.exp(real_score))
            row =(videos[idx], paths[idx], labels.item(), index.item(), loss.item(),
                  outputs[0, 0].item(), outputs[0, 1].item(), fake_prob)
            rows.append(row)
            # print statistics
            running_loss += loss.item()

        # Print results at the end of the epoch
        print('Finished Testing prediction of Model.')
        print('Total execution', running_loss / idx)

    res = pd.DataFrame(rows, columns = ('video', 'path', 'label', 'predicted', 'loss', 'score_real', 'score_fake', 'fake_prob'))
    res.to_csv(path, index = False)