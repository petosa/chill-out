import torchvision.models as models
from torch.utils.model_zoo import load_url
from torch import nn


# Generate a fresh AlexNet model with the given number of output classes.
def load_alexnet(num_classes, pretrained=True, cuda=True):
    model = models.AlexNet()
    if pretrained:
        url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
        state_dict = load_url(url, progress=True)
        model.load_state_dict(state_dict)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
    model = model.cuda() if cuda else model
    return model

# Generate a fresh SqueezeNet model with the given number of output classes.
def load_squeezenet(num_classes, pretrained=True, cuda=True):
    model = models.SqueezeNet(version=1.1)
    if pretrained:
        url = "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth"
        state_dict = load_url(url, progress=True)
        model.load_state_dict(state_dict)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model = model.cuda() if cuda else model
    return model