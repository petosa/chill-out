from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
import torch.nn as nn

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 0
            nn.ReLU(inplace=True), # 1
            nn.MaxPool2d(kernel_size=3, stride=2), # 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2), # 3
            nn.ReLU(inplace=True), # 4
            nn.MaxPool2d(kernel_size=3, stride=2), # 5
            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 6
            nn.ReLU(inplace=True), # 7
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 8
            nn.ReLU(inplace=True), # 9
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 10
            nn.ReLU(inplace=True), # 11
            nn.MaxPool2d(kernel_size=3, stride=2), # 12
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) #13
        self.classifier = nn.Sequential(
            nn.Dropout(), # 0
            nn.Linear(256 * 6 * 6, 4096), # 1
            nn.ReLU(inplace=True), # 2
            nn.Dropout(), # 3 
            nn.Linear(4096, 4096), # 4
            nn.ReLU(inplace=True), # 5
            nn.Linear(4096, num_classes), # 6
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, num_classes=10, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    
    #set_parameter_requires_grad(model_ft, feature_extract)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
    input_size = 224
    return model