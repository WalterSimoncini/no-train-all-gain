# FUNGI HummingBird Evaluation

[Walter Simoncini](https://walter.ashita.nl/)<sup>1</sup>, [Andrei Bursuc](https://abursuc.github.io/)<sup>2</sup>, [Spyros Gidaris](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This repository contains the in-context scene understanding experiments for our paper [No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations](https://fungi.ashita.nl/). This repository is derived from the evaluation framework of [Pariza et al.](https://github.com/vpariza/open-hummingbird-eval). We recommend using the original repository to evaluate the raw DINO features.

## Setup

Set up the repository by creating an appropriate conda environment and installing the dependencies with the following commands:

```sh
conda create -n hummingbird-fungi python=3.10
conda activate hummingbird-fungi

pip install -r requirements.txt
```

Once you've setup your environment download and unpack the full Pascal VOC dataset using the following command:

```sh
wget -O voc_data.zip "https://c2ymfq.am.files.1drv.com/y4mZqATCHOv_Z88obTJ_ZatMGDEx6ts5TzOyJnVLKqmkXZwdL_PKIMFLNmZR9FLFJ1CHMC6h_7bJjxlcUM8yXjG92Ms1X-95x6Dh90QgawSpYDPoE1gmLx3VwnW2amZZEog-omFd87fTKZqn3lpP_mtisDQsfDzruBgz_JHcSWsZd2jrsN2qoV3cJ5HammjY_im0ftfjwNOup1EuiWcQ9KT6A"
unzip voc_data.zip
```

You can also download a tiny version of Pascal VOC from [this link](https://cpf5rw.am.files.1drv.com/y4mxW5pUDPP2WOVWXZYdhd5PK82qXcqQURFvVTXGSfEk8igihjx8oyA_iSmeuMnwLSyNwN601Jq9Ec9PyN3olpCCCoNtrgYg7DkvrXUXyI-mymjQiSR0kgRZDvRRUw4SD9i4QFTa-q5W_A7WahXU5v1XroWUU8bKDgUbY0xbBuX67xZr4HGDQiT5b0cr4iPQbt6NJlKkAyWdXUkIL37xpL3JQ).

## Running Experiments

Each experiment is split into two parts: the creation of a support index, used to fetch the negative batch for the SimCLR loss, and the evaluation itself. Assuming the Pascal VOC dataset is stored in the `VOCSegmentation` folder, you can build the supporting index using the commands below. You should then be able to replicate the second row, first column of Table 3 in the paper. This document also reports the results for the scripts below on `VOC-Tiny`.

```sh
# Create a directory to store the supporting index
# features, labels and the ScaNN index itself
BASE_DIR=support_index

mkdir -p $BASE_DIR/index

SEED=42
# Change depending on your GPU memory
BATCH_SIZE=32
PATCH_SIZE=16
IMAGE_SIZE=512
# Pick a DINO backbone between the ones available here
# https://github.com/facebookresearch/dino
BACKBONE=dino_vits16
AUGMENTATION_EPOCHS=1
# Path to Pascal VOC
DATASET_DIRECTORY=./VOCSegmentation
# The size of the support index. We use the same memory bank
# size for both the "real" memory bank and the support index.
# Pick one between (102400, 1024000, 10240000)
MEMORY_BANK_SIZE=102400

# ScaNN index parameters (we use 5 neighbors as
# the loss computation only requires 2)
NUM_NEIGHBORS=5
NUM_LEAVES=512
NUM_LEAVES_TO_SEARCH=32
NUM_RERANK_CANDIDATES=120
ANISOTROPIC_QUANTIZATION_THRESHOLD=0.2
DIMENSIONS_PER_BLOCK=4

python create_memory_bank.py \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --input-size $IMAGE_SIZE \
    --data-dir $DATASET_DIRECTORY \
    --model $BACKBONE \
    --memory-bank $BASE_DIR/features.pth \
    --memory-bank-labels $BASE_DIR/labels.pth \
    --scann-index $BASE_DIR/index \
    --memory-size $MEMORY_BANK_SIZE \
    --patch-size $PATCH_SIZE \
    --augmentation-epochs $AUGMENTATION_EPOCHS \
    --num-neighbors $NUM_NEIGHBORS \
    --num-leaves $NUM_LEAVES \
    --num-leaves-to-search $NUM_LEAVES_TO_SEARCH \
    --num-reordering-candidates $NUM_RERANK_CANDIDATES \
    --anisotropic-quantization-threshold $ANISOTROPIC_QUANTIZATION_THRESHOLD \
    --dimensions-per-block $DIMENSIONS_PER_BLOCK
```

Once you've created the supporting index, use the following commands to run the evaluation:

```sh
SEED=42
BATCH_SIZE=4
PATCH_SIZE=16
IMAGE_SIZE=512
# The size of the projection head output
LATENT_DIM=96
# Pick a DINO backbone between the ones available here
# https://github.com/facebookresearch/dino
BACKBONE=dino_vits16
AUGMENTATION_EPOCHS=1
# Path to Pascal VOC
DATASET_DIRECTORY=./VOCSegmentation
# Number of per-patch negatives
NUM_NEIGHBORS=2
# The size of the memory bank. Should match the support index size.
# Pick one between (102400, 1024000, 10240000)
MEMORY_BANK_SIZE=102400
# The number of patch negatives that are used for the loss computation
NEGATIVE_BATCH_PERCENT=0.5
# Path to a folder with the support index features and NN index
SUPPORT_DIR=support_index
# Path to the layer used to extract gradients
LAYER_PATH=backbone.blocks.11.attn.proj
# Cross-attention temperature
TEMPERATURE=0.02

# ScaNN index parameters
NUM_NEIGHBORS=30
NUM_LEAVES=512
NUM_LEAVES_TO_SEARCH=32
NUM_RERANK_CANDIDATES=120
ANISOTROPIC_QUANTIZATION_THRESHOLD=0.2
DIMENSIONS_PER_BLOCK=4

python eval_gradients.py \
    --seed $SEED \
    --use-fp16 \
    --batch-size $BATCH_SIZE \
    --input-size $IMAGE_SIZE \
    --patch-size $PATCH_SIZE \
    --gradients-layer $LAYER_PATH \
    --data-dir $DATASET_DIRECTORY \
    --model $BACKBONE \
    --n-neighbors $NUM_NEIGHBORS \
    --memory-bank $SUPPORT_DIR/features.pth \
    --scann-index $SUPPORT_DIR/index \
    --memory-size $MEMORY_BANK_SIZE \
    --latent-dim $LATENT_DIM \
    --negative-batch-percent $NEGATIVE_BATCH_PERCENT \
    --temperature $TEMPERATURE \
    --num-neighbors $NUM_NEIGHBORS \
    --num-leaves $NUM_LEAVES \
    --num-leaves-to-search $NUM_LEAVES_TO_SEARCH \
    --num-reordering-candidates $NUM_RERANK_CANDIDATES \
    --anisotropic-quantization-threshold $ANISOTROPIC_QUANTIZATION_THRESHOLD \
    --dimensions-per-block $DIMENSIONS_PER_BLOCK
```

Using `VOC-Tiny` this experiment should result in a mIoU of 46.80%.

### Data Efficient Scene Understanding

The experiments for data-efficient (i.e., with a limited number of samples) in-context scene understanding can be run using very similar commands as the regular experiment. Once again, start by creating the support index using the commands below (we use 83 training samples here).

It's important to note that we do not specify a memory bank size, leaving it unbounded, as the original memory bank size for this experiment (20480000) allows us to store the entirety of the dataset, as in the worst case we have:

$N * P * A = 165 \times 1024 \times 8 = 1351680 < 20480000$

patches to store, where:

- $N$: number of images in the training dataset
- $P$: number of patches per image
- $A$: number of augmentation epochs

```sh
# Create a directory to store the supporting index
# features, labels and the ScaNN index itself
BASE_DIR=support_index_efficient

mkdir -p $BASE_DIR/index

SEED=42
# Change depending on your GPU memory
BATCH_SIZE=32
PATCH_SIZE=16
IMAGE_SIZE=512
# Pick a DINO backbone between the ones available here
# https://github.com/facebookresearch/dino
BACKBONE=dino_vits16
AUGMENTATION_EPOCHS=8
# Path to Pascal VOC
DATASET_DIRECTORY=./VOCSegmentation

# ScaNN index parameters (we use 5 neighbors as the
# loss computation only requires 2)
NUM_NEIGHBORS=5
NUM_LEAVES=512
NUM_LEAVES_TO_SEARCH=32
NUM_RERANK_CANDIDATES=120
ANISOTROPIC_QUANTIZATION_THRESHOLD=0.2
DIMENSIONS_PER_BLOCK=4

# Training dataset size
TRAIN_SAMPLES=83

python create_memory_bank.py \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --input-size $IMAGE_SIZE \
    --data-dir $DATASET_DIRECTORY \
    --model $BACKBONE \
    --memory-bank $BASE_DIR/features.pth \
    --memory-bank-labels $BASE_DIR/labels.pth \
    --scann-index $BASE_DIR/index \
    --patch-size $PATCH_SIZE \
    --augmentation-epochs $AUGMENTATION_EPOCHS \
    --num-neighbors $NUM_NEIGHBORS \
    --num-leaves $NUM_LEAVES \
    --num-leaves-to-search $NUM_LEAVES_TO_SEARCH \
    --num-reordering-candidates $NUM_RERANK_CANDIDATES \
    --anisotropic-quantization-threshold $ANISOTROPIC_QUANTIZATION_THRESHOLD \
    --dimensions-per-block $DIMENSIONS_PER_BLOCK \
    --num-train-samples $TRAIN_SAMPLES
```

Once you've created the support index, use the script below to run the evaluation.

```sh
SEED=42
BATCH_SIZE=4
PATCH_SIZE=16
IMAGE_SIZE=512
# The size of the projection head output
LATENT_DIM=96
# Pick a DINO backbone between the ones available here
# https://github.com/facebookresearch/dino
BACKBONE=dino_vits16
AUGMENTATION_EPOCHS=8
# Path to Pascal VOC
DATASET_DIRECTORY=./VOCSegmentation
# Number of per-patch negatives
NUM_NEGATIVE_NEIGHBORS=2
# The number of patch negatives that are used for the loss computation
NEGATIVE_BATCH_PERCENT=0.5
# Path to a folder with the support index features and NN index
SUPPORT_DIR=support_index_efficient
# Path to the layer used to extract gradients
LAYER_PATH=backbone.blocks.11.attn.proj
# Cross-attention temperature
TEMPERATURE=0.1

# ScaNN index parameters
NUM_NEIGHBORS=90
NUM_LEAVES=512
NUM_LEAVES_TO_SEARCH=256
NUM_RERANK_CANDIDATES=1800
ANISOTROPIC_QUANTIZATION_THRESHOLD=0.2
DIMENSIONS_PER_BLOCK=4

# Training dataset size
TRAIN_SAMPLES=83

python eval_gradients.py \
    --seed $SEED \
    --use-fp16 \
    --batch-size $BATCH_SIZE \
    --input-size $IMAGE_SIZE \
    --patch-size $PATCH_SIZE \
    --gradients-layer $LAYER_PATH \
    --data-dir $DATASET_DIRECTORY \
    --model $BACKBONE \
    --num-negative-neighbors $NUM_NEGATIVE_NEIGHBORS \
    --memory-bank $SUPPORT_DIR/features.pth \
    --scann-index $SUPPORT_DIR/index \
    --latent-dim $LATENT_DIM \
    --negative-batch-percent $NEGATIVE_BATCH_PERCENT \
    --temperature $TEMPERATURE \
    --num-neighbors $NUM_NEIGHBORS \
    --num-leaves $NUM_LEAVES \
    --num-leaves-to-search $NUM_LEAVES_TO_SEARCH \
    --num-reordering-candidates $NUM_RERANK_CANDIDATES \
    --anisotropic-quantization-threshold $ANISOTROPIC_QUANTIZATION_THRESHOLD \
    --dimensions-per-block $DIMENSIONS_PER_BLOCK \
    --num-train-samples $TRAIN_SAMPLES
```

Using `VOC-Tiny` this experiment should result in a mIoU of 27.69%.
