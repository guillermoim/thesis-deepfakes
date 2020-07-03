import networks as ns
import torch
import dataset as ds
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    '''
        This has CrossEntropy, SGD, y scheduler every 10 epochs. 
        Simulation using:
         - EfficientNet-B3
         - CrossEntropy Loss with weights
         - Adam
         - Epoch size 2500, 30 epochs, scheduler every 10, batch size = 4x48
    '''
    model_name = 'efficientnet-b3'
    width, depth, resize, dropout = utils.efficientnet_params(model_name)

    # blocks_args, global_params = utils.efficientnet(width_coefficient=width, depth_coefficient=depth, image_size=res,
    #             dropout_rate=dropout,  num_classes=2)

    model = EfficientNet.from_pretrained(model_name,
                                         num_classes=2)  # blocks_args=blocks_args, global_params=global_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")

    model = torch.nn.DataParallel(model)

    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    dataset = ds.AugmentedDataset('datasets/full_train_dataset.csv', 'data/faces/', transform = transform, p = .75)

    exec_name = 'EffNet-B3 Single - Variant 1'
    epochs = 30
    batch_size = 48
    epoch_size = 2500

    print('\t\t Training Details')
    print('------------------------------------')

    print(f'Execution name: {exec_name}')
    print(f'Epochs: {epochs}')
    print(f'Epoch size: {epoch_size}')
    print(f'Batch size: {batch_size}')
    print(f'Optimizer Adam')
    print(f'Power StepLR: 10 epochs')

    # TRAIN
    print('\t\t Epoch information - Single Model')
    print('------------------------------------')

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=.5, step_size=10)

    ns.train_var_2(exec_name, model, dataset,
                   criterion, optimizer, scheduler,
                   epochs, batch_size, epoch_size, device)

    path_to_save = 'models/effnet_b3-v2-1.pth'
    torch.save(model.state_dict(), path_to_save)