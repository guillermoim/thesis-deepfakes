import networks as ns
import torch
import dataset as ds
from efficientnet_pytorch import EfficientNet

if __name__ == '__main__':

    model = ns.CustomEnsemble('efficientnet-b0', num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)
    model.to(device)

    path_to_save = 'data/ens_effnet_b0.pth'

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load('data/ens_effnet_b0.pth'))

    dataset = ds.BalancedClusterDataset('datasets/ob_dataset.csv')

    # TRAIN
    print('\t\t Training information')
    print('------------------------------------')
    ns.train('eff-net b0', model, dataset, 30, 64, device)
    torch.save(model.state_dict(), path_to_save)

    model.eval()
    print('\n\t\t Test information')
    print('------------------------------------')
    path = 'test_execution.csv'
    ns.test(model, dataset, device, path)