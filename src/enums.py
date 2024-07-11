from enum import Enum


class ModelType(Enum):
    MOCO_V3_VIT_B_16 = "vit-b16-mocov3"

    MAE_VIT_B_16 = "mae-vit-b16"

    DINO_VIT_16_B = "dino_vitb16"
    DINO_V2_VIT_14_B = "dinov2_vitb14"

    CLIP_VIT_16_B = "clip-vit-b16"
    EVA_CLIP_VIT_16_B = "eva-clip-vit-b16"

    VIT_S_16 = "vit-s16"
    VIT_B_16 = "vit-b16"
    VIT_B_32 = "vit-b32"

    VIT_B_16_AUGREG_IN1K = "vit-b16-augreg-in1k"

    VIT_S_16_AUGREG_IN21K = "vit-s16-augreg-in21k"
    VIT_B_16_AUGREG_IN21K = "vit-b16-augreg-in21k"
    VIT_L_16_AUGREG_IN21K = "vit-l16-augreg-in21k"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DatasetType(Enum):
    IN1K = "imagenet-1k"
    IMAGENET_100 = "imagenet-100"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TEXTURES = "textures"
    FLOWERS_102 = "flowers102"
    STANFORD_CARS = "cars"
    FOOD_101 = "food101"
    FGVC_AIRCRAFT = "aircraft"
    OXFORD_IIT_PETS = "pets"
    CUB_200_2011 = "cub-200"
    EUROSAT = "eurosat"

    PARIS_LANDMARKS = "paris-landmarks"
    OXFORD_LANDMARKS = "oxford-landmarks"
