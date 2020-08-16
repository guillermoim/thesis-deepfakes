import torch
from torch.nn import DataParallel
from timm.models import efficientnet
from pretrainedmodels.models.inceptionv4 import inceptionv4
from pretrainedmodels.models.xception import xception, pretrained_settings
from efficientnet_pytorch import EfficientNet
from RegNet import regnety
from focal_loss import FocalLoss

models_info = {'efficientnet-b3': {
                    'normalization': {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    'input_resize': 300,
                    'num_features' : 1536,
                },
               'efficientnet-b5': {
                    'normalization': {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    'input_resize' : 456,
                    'num_features' : 2048,

               },
               'efficientnet-b7': {
                    'normalization': {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    'input_resize': 600,
                    'num_features' : 2560,
               },
               'xception' : {
                    'normalization': {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                    'input_resize' : 299,
                    'num_features' : 2048,
               },
               'inceptionv4':{
                    'normalization': {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                    'input_resize' : 299,
                    'num_features' : 1536,
               },
               'regnety-1.6GF':{
                    'normalization': {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    'input_resize': 224,
                    'num_features': 888,
                },
            'regnety-8.0GF':{
                    'normalization': {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                    'input_resize': 224,
                    'num_features': 2016,
                }
            }


class  HelperModel(torch.nn.Module):
    def __init__(self, encoder, num_features, out_classes, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(num_features, out_classes, bias=False)


    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_saved_model(path_to_model):

    saved = torch.load(path_to_model)

    model, resize, _, _, _, normalization, _ = load_config(saved['model_name'], saved['variant'], 10, 10)

    model = DataParallel(model).cuda()

    model.load_state_dict(saved['model_state_dict'])

    return model, resize, saved['model_name'], saved['best_loss'], saved['epoch'], normalization

def load_config(name, variant, n_epochs, epoch_size):

    # Just use BCE & FocalLoss
    num_classes = 1

    if name == 'efficientnet-b3':

        encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = HelperModel(encoder, models_info[name]['num_features'], num_classes)

    if name == 'efficientnet-b5':

        encoder = efficientnet.tf_efficientnet_b5_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = HelperModel(encoder, models_info[name]['num_features'], num_classes)

    if name == 'efficientnet-b7':

        encoder = efficientnet.tf_efficientnet_b7_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = HelperModel(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'xception':

        encoder = xception()
        encoder.forward_features = encoder.features
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = HelperModel(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'inceptionv4':

        encoder = inceptionv4()
        encoder.forward_features = encoder.features
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = HelperModel(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'regnety-1.6GF':

        model = regnety('1.6GF')
        model.head.fc = torch.nn.Linear(models_info[name]['num_features'], num_classes, bias=False)
        resize = models_info[name]['input_resize']
        normalization = models_info[name]['normalization']

    elif name == 'regnety-8.0GF':

        model = regnety('8.0GF')
        model.head.fc = torch.nn.Linear(models_info[name]['num_features'], num_classes, bias=False)
        resize = models_info[name]['input_resize']
        normalization = models_info[name]['normalization']

    if variant == 0:

        lr = 0.01

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda x : (n_epochs*epoch_size - x) / (n_epochs*epoch_size)),
                     'mode': 'iteration'}

        desc = {'model_name' : name,
                'epochs' : str(n_epochs),
                'epoch_size': str(epoch_size),
                'Loss function': 'BCE-No weighted',
                'optim': 'SGD + momentum=.9 + weight_decay=1e-4 + nesterov',
                'initial_lr': str(lr),
                'secheduler': 'lambda with iteration step'}


    elif variant == 1:

        lr = 0.0002

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=6),
                     'mode': 'epoch'}

        desc = {'model_name' : name,
                'epochs' : str(n_epochs),
                'epoch_size': str(epoch_size),
                'Loss function': 'No weight', #BCE-1*real+0.75*fake',
                'optim': 'Default Adam',
                'initial_lr': str(lr),
                'secheduler': '0.1 by epoch'}

    else:

        lr = 0.005

        # Implement Focal-Loss
        criterion = FocalLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (n_epochs * epoch_size - x) / (n_epochs * epoch_size)),
                     'mode': 'iteration'}

        desc = {'model_name': name,
                'epochs': str(n_epochs),
                'epoch_size': str(epoch_size),
                'Loss function': 'FocalLoss',
                'optim': 'SGD + momentum=.9 + weight_decay=1e-4 + nesterov',
                'initial_lr': str(lr),
                'secheduler': 'lambda with iteration step'}

    return model, resize, criterion, optimizer, scheduler, normalization, desc


def get_available_models():
    return models_info.keys()

