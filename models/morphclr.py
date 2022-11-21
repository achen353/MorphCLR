import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class MorphCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(MorphCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
