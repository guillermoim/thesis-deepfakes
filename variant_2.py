import networks as ns
import torch
import dataset as ds
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    '''
        This has BinaryCrossEntropy, SGD, y scheduler every 10 epochs. 
    '''

    model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")

    model = torch.nn.DataParallel(model)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    dataset = ds.AugmentedDataset('datasets/full_train_dataset.csv', 'data/faces/', transform = transform)

    exec_name = 'EffNet-B7 Single - Variant 2'
    epochs = 30
    batch_size = 48
    epoch_size = 2500

    print('\t\t Training Details')
    print('------------------------------------')

    print(f'Execution name: {exec_name}')
    print(f'Epochs: {epochs}')
    print(f'Epoch size: {epoch_size}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer SGD')
    print(f'Power StepLR: 10 epochs')

    # TRAIN
    print('\t\t Epoch information - Single Model')
    print('------------------------------------')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    ns.train_var_2(exec_name, model, dataset,
                   criterion, optimizer, scheduler,
                   epochs, batch_size, epoch_size, device)

    path_to_save = 'models/effnet_b7-v2-2.pth'
    torch.save(model.state_dict(), path_to_save)