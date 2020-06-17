import networks as ns
import torch
import dataset as ds
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':

    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model.to(device)



    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    model = torch.nn.DataParallel(model)

    dataset = ds.AugmentedDataset('datasets/full_train_dataset.csv', 'data/faces/')

    exec_name = 'EffNet-B7'
    epochs = 5
    batch_size = 48
    epoch_size = 200
    power = 1

    print('\t\t Training Details')
    print('------------------------------------')

    print(f'Execution name: {exec_name}')
    print(f'Epochs: {epochs}')
    print(f'Epoch size: {epoch_size}')
    print(f'Batch size: {batch_size}')
    print(f'Power PolyLR: {power}')

    # TRAIN
    print('\t\t Epoch information - Single Model')
    print('------------------------------------')
    ns.train_var_2(exec_name, model, dataset, epochs, batch_size, epoch_size, power, device)

    path_to_save = 'models/effnet_b7-v2.pth'
    torch.save(model.state_dict(), path_to_save)
