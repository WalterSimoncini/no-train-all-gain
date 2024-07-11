import timm
import torch.nn as nn

from .base_factory import ModelFactory
 

class MAEViT16BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_base_patch16_224.mae", pretrained=True)

    def get_transform(self, **kwargs) -> nn.Module:
        config = timm.data.resolve_model_data_config(
            timm.create_model("vit_base_patch16_224.mae", pretrained=True)
        )

        return timm.data.create_transform(**config, is_training=False)
