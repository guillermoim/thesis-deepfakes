import networks as ns
import torch
import dataset as ds
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':
    name_sim = 'ens_effnet_b3'

    model = ns.CustomEnsemble('efficientnet-b3', num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)
    model.to(device)

    path_to_save = f'models/{name_sim}.pth'

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    model = torch.nn.DataParallel(model)

    dataset = ds.BalancedClusterDataset('datasets/ob_dataset.csv')

    # TRAIN
    print('\t\t Training information')
    print('------------------------------------')
    ns.train(name_sim, model, dataset, 25, 32, device)
    torch.save(model.state_dict(), path_to_save)

    model.load_state_dict(torch.load(path_to_save))
    model.eval()
    print('\n\t\t Test information')
    print('------------------------------------')
    date_time = datetime.now().strftime('%Y-%M-%d_%H_%m')
    path = f'outputs/test_results_{date_time}.csv'
    ns.test(model, dataset, device, path)