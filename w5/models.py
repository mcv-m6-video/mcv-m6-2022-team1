from torch import nn
from torch.nn.functional import normalize
from torchvision.models import resnet


class LinearEncoder(nn.Module):
    def __init__(self, input_size: int, layer_sizes: list):
        super(LinearEncoder, self).__init__()
        layers = [nn.Linear(input_size, layer_sizes[0])]

        for ii in range(len(layer_sizes) - 1):
            layers.append(nn.ReLU())
            # layers.append(Dropout(0.5))
            # layers.append(LayerNorm(layer_sizes[ii]))
            layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii + 1]))

        self.linear = nn.Sequential(*layers)

    def init_weights(self):
        # Linear
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(
                    layer.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.linear(x)
        x = normalize(x, p=2.0, dim=-1)
        return x


class CarIdResnet(nn.Module):
    def __init__(self, layer_sizes: list):
        super(CarIdResnet, self).__init__()
        backbone = resnet.resnet18(pretrained=True, progress=True)
        embedder = LinearEncoder(512, layer_sizes)
        self.model = nn.Sequential(*list(backbone.children())[:-1], embedder)

    def freeze(self, to_freeze: int):
        if to_freeze < 0:
            return
        model_children = list(self.model.children())
        index = len(model_children) - to_freeze
        for layer in model_children[:index]:
            layer.requires_grad_(False)

    def forward(self, x):
        return self.model(x)
