import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class MorphCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(MorphCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)

        # Freeze gradients of the resnet for extra sanity
        for x in self.backbone.parameters(): x.requires_grad = False

        dim_mlp = self.backbone.fc.out_features

        self.test_variable = nn.Parameter(torch.randn(dim_mlp))


        

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model
    def get_parameters(self):
        return [self.test_variable,]
    def forward(self, x):
        print("#### test variable ####\n", self.test_variable)
        print("#### backbone variable ####\n", self.backbone.fc.weight)
        return self.backbone(x) + self.test_variable[None, :]
