import torchvision.models as models
from torch.utils.model_zoo import load_url
from torch import nn


def get_trainable_layers(model):
    layers = []
    for _, w in model.named_children():
        for _, w1 in w.named_children():
            #These loops go over all nested children in the model architecture (including those without grad updates)
            for l, _ in w1.named_parameters():
                #This loop filters out any children that aren't trainable, but we only want to count the layer if theres at least 1 trainable node within it.
                if w1 not in layers:
                    layers.append(w1)
    return layers


def load_alexnet(num_classes):
    url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    model = models.AlexNet()
    state_dict = load_url(url, progress=True)
    model.load_state_dict(state_dict)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
    return model

def load_squeezenet(num_classes):
    url = "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth"
    model = models.SqueezeNet()
    state_dict = load_url(url, progress=True)
    model.load_state_dict(state_dict)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model
    