import torch
from timm.models import efficientnet
from pretrainedmodels.models.inceptionv4 import inceptionv4
from pretrainedmodels.models.xception import xception, pretrained_settings
from efficientnet_pytorch import EfficientNet
from RegNet import regnety

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
                }
            }


class Model(torch.nn.Module):
    def __init__(self, encoder, num_features, out_classes, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(num_features, out_classes, bias=True)


    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_saved_model(path_to_mode, model_name):
    # TODO: Given name and path to model, load saved model
    pass

def load_config(name, variant, n_epochs, epoch_size):

    num_classes = 1 if variant < 1 else 2

    cast = 'torch.LongTensor' if num_classes == 1 else 'torch.FloatTensor'


    if name == 'efficientnet-b3':

        encoder = efficientnet.tf_efficientnet_b3_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = Model(encoder, models_info[name]['num_features'], num_classes)

    if name == 'efficientnet-b5':

        encoder = efficientnet.tf_efficientnet_b5_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = Model(encoder, models_info[name]['num_features'], num_classes)

    if name == 'efficientnet-b7':

        encoder = efficientnet.tf_efficientnet_b7_ns(pretrained=True, drop_path_rate=0.2)
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = Model(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'xception':

        encoder = xception()
        encoder.forward_features = encoder.features
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = Model(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'inceptionv4':

        encoder = inceptionv4()
        encoder.forward_features = encoder.features
        normalization = models_info[name]['normalization']
        resize = models_info[name]['input_resize']
        model = Model(encoder, models_info[name]['num_features'], num_classes)

    elif name == 'regnety-1.6GF':

        model = regnety('1.6GF')
        model.head.fc = torch.nn.Linear(models_info[name]['num_features'], num_classes)
        resize = models_info[name]['input_resize']
        normalization = models_info[name]['normalization']

    if variant == 0:

        '''
            Variant 0 has no weighted BCE, with SGD and epoch-updated LR.
        '''

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.75).cuda())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda x : (n_epochs*epoch_size - x) / (n_epochs*epoch_size)),
                     'mode': 'iteration'}

    elif variant == 1:

        '''
            Variant 1 has weighted BCE (1 * real + 0.75 * fake), with SGD and Lambda scheduler 
        '''

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.75).cuda())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size= n_epochs // 3),
                     'mode': 'epoch'}
    else:
        # Implement Focal-Loss
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.75).cuda())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3),
                     'mode': 'epoch'}

    return model, resize, criterion, optimizer, scheduler, normalization, cast


def get_available_models():
    return models_info.keys()

