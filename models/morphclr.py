
import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class MorphCLRSingle(nn.Module):
    def __init__(self, base_model, out_dim, use_pretrained):
        super(MorphCLRSingle, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=use_pretrained),
            "resnet50": models.resnet50(pretrained=use_pretrained),
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


class MorphCLRDual(nn.Module):
    def __init__(self, base_model, out_dim, use_pretrained):
        super(MorphCLRDual, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=use_pretrained),
            "resnet50": models.resnet50(pretrained=use_pretrained),
        }

        self.backbone_1 = self._get_basemodel(base_model)
        self.backbone_2 = self._get_basemodel(base_model)
 
        dim_mlp = self.backbone_1.fc.in_features
        self.backbone_1.fc = nn.Identity()
        self.backbone_2.fc = nn.Identity()

        self.proj_layer = nn.Sequential(
            nn.Linear(dim_mlp * 2, dim_mlp * 2),
            nn.ReLU(),
            nn.Linear(dim_mlp * 2, out_dim)
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
        # x is a stacked tensor of shape [2, batch_size, channel, width, length]
        x_1, x_2 = x[0], x[1]
        x_1, x_2 = self.backbone_1(x_1), self.backbone_2(x_2)
        x_cat = torch.cat([x_1, x_2], 1)
        x_proj = self.proj_layer(x_cat)
        return x_proj
