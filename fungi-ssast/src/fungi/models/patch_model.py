import os
import requests
import torch.nn as nn

from src.models import ASTModel
from .base_factory import ModelFactory


class SSASTPatchFactory(ModelFactory):
    def build(self, cache_dir: str, input_tdim: int = 512, **kwargs) -> nn.Module:
        checkpoint_name = "SSAST-Base-Patch-400.pth"
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)

        if not os.path.isfile(checkpoint_name):
            # Download the Patch SSAST checkpoint and save it locally
            checkpoint = requests.get(f"https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1")

            with open(checkpoint_path, "wb") as checkpoint_file:
                checkpoint_file.write(checkpoint.content)

        model = ASTModel(
            label_dim=10,
            fshape=16,
            tshape=16,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=input_tdim,
            model_size="base",
            pretrain_stage=False,
            load_pretrained_mdl_path=checkpoint_path
        )

        model.mlp_head = nn.Identity()

        return model
