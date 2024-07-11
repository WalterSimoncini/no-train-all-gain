import os
import timm
import torch
import logging
import requests
import torch.nn as nn
import torchvision.transforms.v2 as tf

from .base_factory import ModelFactory
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights


class ViT16BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()

        return model

    def get_transform(self, **kwargs) -> nn.Module:
        return ViT_B_16_Weights.DEFAULT.transforms()


class ViT32BFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        model.heads = nn.Identity()

        return model

    def get_transform(self, **kwargs) -> nn.Module:
        return ViT_B_32_Weights.DEFAULT.transforms()


class ViT16SFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_small_patch16_224.augreg_in1k", pretrained=True, num_classes=0)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_small_patch16_224.augreg_in1k"),
            is_training=False
        )


class ViT16BAugRegIN1KFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=0)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_base_patch16_224.augreg_in1k"),
            is_training=False
        )


class ViT16SAugRegIN21KFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_small_patch16_224.augreg_in21k", pretrained=True, num_classes=0)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_small_patch16_224.augreg_in21k"),
            is_training=False
        )


class ViT16BAugRegIN21KFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_base_patch16_224.augreg_in21k"),
            is_training=False
        )


class ViT16LAugRegIN21KFactory(ModelFactory):
    def build(self, **kwargs) -> nn.Module:
        return timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True, num_classes=0)

    def get_transform(self, **kwargs) -> nn.Module:
        return timm.data.create_transform(
            **timm.data.resolve_model_data_config("vit_large_patch16_224.augreg_in21k"),
            is_training=False
        )


class MoCoV3ViT16BFactory(ModelFactory):
    def build(self, cache_dir: str, **kwargs) -> nn.Module:
        checkpoint_name = "vit-b-300ep.pth.tar"
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)

        if not os.path.isfile(checkpoint_name):
            # Download the MoCo V2 checkpoint and save it locally
            checkpoint = requests.get(f"https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/{checkpoint_name}")

            with open(checkpoint_path, "wb") as checkpoint_file:
                checkpoint_file.write(checkpoint.content)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cpu")
        )["state_dict"]

        # Taken from https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py
        for k in list(checkpoint.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith("module.base_encoder.fc"):
                # remove prefix
                checkpoint[k[len("module.base_encoder."):]] = checkpoint[k]

            # delete renamed or unused k
            del checkpoint[k]

        model = timm.create_model("vit_base_patch16_224.augreg_in1k", num_classes=0)
        msg = model.load_state_dict(checkpoint, strict=False)

        logging.info(f"loaded MoCo v3 ViT 16/B from checkpoint: {msg}")

        return model

    def get_transform(self, **kwargs) -> nn.Module:
        # Taken from https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py#L288
        return tf.Compose([
            tf.Resize(256),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
