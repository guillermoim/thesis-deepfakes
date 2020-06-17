import torch
from efficientnet_pytorch import EfficientNet

class CustomEnsemble(torch.nn.Module):

    def __init__(self, name, num_classes):
        super(CustomEnsemble, self).__init__()
        torch.cuda.manual_seed(10)
        self.m0 = EfficientNet.from_pretrained(name, num_classes=num_classes)
        torch.cuda.manual_seed(80131)
        self.m1 = EfficientNet.from_pretrained(name, num_classes=num_classes)
        torch.cuda.manual_seed(9183120)
        self.m2 = EfficientNet.from_pretrained(name, num_classes=num_classes)
        torch.cuda.manual_seed(1231)
        self.m3 = EfficientNet.from_pretrained(name, num_classes=num_classes)

    def forward(self, x):

        y0 = self.m0(x.clone()).unsqueeze(0)
        y1 = self.m1(x.clone()).unsqueeze(0)
        y2 = self.m2(x.clone()).unsqueeze(0)
        y3 = self.m3(x.clone()).unsqueeze(0)

        y = torch.cat((y0, y1, y2, y3))
        mean =  torch.mean(y, dim=0)

        return mean
