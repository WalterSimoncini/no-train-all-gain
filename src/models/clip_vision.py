import timm
import torch.nn as nn

from .base_factory import ModelFactory
 

class CLIPViT16BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)

    def get_transform(self, **kwargs) -> nn.Module:
        config = timm.data.resolve_model_data_config(
            timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True)
        )

        return timm.data.create_transform(**config, is_training=False)


class EVACLIPViT16BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("eva02_base_patch16_clip_224.merged2b", pretrained=True)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("eva02_base_patch16_clip_224.merged2b"),
            is_training=False
        )
