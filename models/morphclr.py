import torch
import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class MorphCLRBase(nn.Module):
    def __init__(self, use_pretrained):
        super().__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=use_pretrained),
            "resnet50": models.resnet50(pretrained=use_pretrained),
        }

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model


class MorphCLRSingle(MorphCLRBase):
    def __init__(self, base_model, out_dim, use_pretrained):
        super().__init__(use_pretrained)

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        return self.backbone(x)


class MorphCLRDual(MorphCLRBase):
    def __init__(self, base_model, out_dim, use_pretrained):
        super().__init__(use_pretrained)
        self.backbone_1 = self._get_basemodel(base_model)
        self.backbone_2 = self._get_basemodel(base_model)

        dim_mlp = self.backbone_1.fc.in_features
        self.backbone_1.fc = nn.Identity()
        self.backbone_2.fc = nn.Identity()

        self.proj_layer = nn.Sequential(
            nn.Linear(dim_mlp * 2, dim_mlp * 2),
            nn.ReLU(),
            nn.Linear(dim_mlp * 2, out_dim),
        )

    def forward(self, x):
        # x is a stacked tensor of shape [2, batch_size, channel, width, length]
        x_1, x_2 = x[0], x[1]
        x_1, x_2 = self.backbone_1(x_1), self.backbone_2(x_2)
        x_cat = torch.cat([x_1, x_2], 1)
        x_proj = self.proj_layer(x_cat)
        return x_proj


class MorphCLRSingleEval(MorphCLRBase):
    def __init__(
        self,
        base_model,
        edge_checkpoint_file_path=None,
        non_edge_checkpoint_file_path=None,
        device="cpu",
        return_embedding=False,
    ):
        super().__init__(use_pretrained=False)
        self.device = device
        self.edge_model = self._get_basemodel(base_model)
        self.non_edge_model = self._get_basemodel(base_model)
        if edge_checkpoint_file_path and non_edge_checkpoint_file_path:
            self._load_checkpoint(
                edge_checkpoint_file_path, non_edge_checkpoint_file_path
            )
        self._init_linear_layer()
        self.to(self.device)
        self.return_embedding = return_embedding

    def _load_checkpoint(
        self, edge_checkpoint_file_path, non_edge_checkpoint_file_path
    ):
        # Load edge checkpoint
        edge_checkpoint = torch.load(
            edge_checkpoint_file_path, map_location=self.device
        )
        edge_state_dict = edge_checkpoint["state_dict"]
        for k in list(edge_state_dict.keys()):
            if k.startswith("backbone."):
                if k.startswith("backbone") and not k.startswith("backbone.fc"):
                    # remove prefix
                    edge_state_dict[k[len("backbone.") :]] = edge_state_dict[k]
            del edge_state_dict[k]
        edge_log = self.edge_model.load_state_dict(edge_state_dict, strict=False)
        assert edge_log.missing_keys == ["fc.weight", "fc.bias"]
        # Load non-edge checkpoint
        non_edge_checkpoint = torch.load(
            non_edge_checkpoint_file_path, map_location=self.device
        )
        non_edge_checkpoint = non_edge_checkpoint["state_dict"]
        for k in list(non_edge_checkpoint.keys()):
            if k.startswith("backbone."):
                if k.startswith("backbone") and not k.startswith("backbone.fc"):
                    # remove prefix
                    non_edge_checkpoint[k[len("backbone.") :]] = non_edge_checkpoint[k]
            del non_edge_checkpoint[k]
        non_edge_log = self.non_edge_model.load_state_dict(
            non_edge_checkpoint, strict=False
        )
        assert non_edge_log.missing_keys == ["fc.weight", "fc.bias"]

    def _init_linear_layer(self):
        dim_mlp = self.edge_model.fc.in_features
        self.edge_model.fc = nn.Identity()
        self.non_edge_model.fc = nn.Identity()
        self.linear = nn.Linear(2 * dim_mlp, 10)

    def forward(self, x):
        edge_x, non_edge_x = x[0], x[1]
        edge_out = self.edge_model(edge_x)
        non_edge_out = self.non_edge_model(non_edge_x)
        x_cat = torch.cat([edge_out, non_edge_out], dim=1)
        out = self.linear(x_cat)

        if self.return_embedding:
            return out, x_cat

        return out


class MorphCLRDualEval(MorphCLRBase):
    def __init__(
        self, base_model, checkpoint_file_path=None, device="cpu", return_embedding=True
    ):
        super().__init__(use_pretrained=False)
        self.device = device
        self.backbone_1 = self._get_basemodel(base_model)
        self.backbone_2 = self._get_basemodel(base_model)
        if checkpoint_file_path:
            self._load_checkpoint(checkpoint_file_path)
        self._init_linear_layer()
        self.to(self.device)
        self.return_embedding = return_embedding

    def _load_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        log = self.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == [
            "backbone_1.fc.weight",
            "backbone_1.fc.bias",
            "backbone_2.fc.weight",
            "backbone_2.fc.bias",
        ]

    def _init_linear_layer(self):
        dim_mlp = self.backbone_1.fc.in_features
        self.backbone_1.fc = nn.Identity()
        self.backbone_2.fc = nn.Identity()
        self.linear = nn.Linear(2 * dim_mlp, 10).to(self.device)

    def forward(self, x):
        edge_x, non_edge_x = x[0], x[1]
        edge_out = self.backbone_1(edge_x)
        non_edge_out = self.backbone_2(non_edge_x)
        x_cat = torch.cat([edge_out, non_edge_out], dim=1)
        out = self.linear(x_cat)

        if self.return_embedding:
            return out, x_cat

        return out
