# FUNGI Features for Text Classification

[Walter Simoncini](https://walter.ashita.nl/)<sup>1</sup>, [Andrei Bursuc](https://abursuc.github.io/)<sup>2</sup>, [Spyros Gidaris](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This repository contains the text classification experiments for our paper [No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations](https://fungi.ashita.nl/).

## Setup

To begin, set up an environment using the following commands:

```sh
conda create -n fungi-text python=3.10
conda activate fungi-text

pip install -r requirements.txt
```

## Running Experiments

The text classification experiments can be split in the following steps:

1. Extract the deep text features
2. Extract the text gradients
3. Evaluate the k-nn classification performance

In the following scripts, we use the `banking-77` dataset, but you can choose any of `trec`, `ag-news`, `banking-77`, `tweet-eval`, `fine-grained-sst`. As for the model, we will be using BERT base (uncased), here labelled as `bert-base-uncased`, but you can also specify `t5-small`.

Note that the embeddings of T5-small are 512-dimensional, while the BERT ones are 768-dimensional.

### Feature Extraction

The script below extracts the features for the training and test sets fo the `banking-77` dataset, using a BERT backbone.

```sh
BATCH_SIZE=64
NUM_WORKERS=18
EMBEDDINGS_DIM=768
DATASET=banking-77
BACKBONE=bert-base-uncased
EMBEDDINGS_DIM=768
OUTPUT_BASE_DIR=./data
OUTPUT_DIR=$OUTPUT_BASE_DIR/features/$DATASET

# Create a directory to store the feature datasets
mkdir -p $OUTPUT_DIR

# Extract features for the train and test sets
python $PWD/extract_features.py \
    --output-path $OUTPUT_DIR/train.h5 \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --dataset $DATASET \
    --embeddings-dim $EMBEDDINGS_DIM \
    --model $BACKBONE \
    --dataset-split train

python $PWD/extract_features.py \
    --output-path $OUTPUT_DIR/test.h5 \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --dataset $DATASET \
    --embeddings-dim $EMBEDDINGS_DIM \
    --model $BACKBONE \
    --dataset-split test
```

## Gradients Extraction

Before extracting gradients, generate a projection matrix using the command below. This assumes you're extracting gradients for a BERT-base backbone, using the `backbone.encoder.layer.11.attention.output.dense` layer, which has `768 * 768 + 768` parameters, including the bias. Adapt the matrix dimensions `(projection_dim, gradients_dim)` as needed.

```sh
python generate_projection.py \
    --projection-dim 768 \
    --gradients-dim 590592 \
    --output-path ./bert-projection.pth
```

### KL Gradients

The script below extracts the KL gradients for the `banking-77` dataset using a BERT backbone.

```sh
SEED=42
BATCH_SIZE=32
EMBEDDINGS_DIM=768
DATASET=banking-77
BACKBONE=bert-base-uncased
OUTPUT_BASE_DIR=./data
OUTPUT_DIR=$OUTPUT_BASE_DIR/kl/$DATASET
PROJECTION_MATRIX=bert-projection.pth

# KL-specific parameters
TEMPERATURE=15
LATENT_DIM=768

# Layer path and HDF5 dataset name (attn_proj)
LAYER_PATH=backbone.encoder.layer.11.attention.output.dense:attn_proj

# Create a directory to store the feature datasets
mkdir -p $OUTPUT_DIR

# Extract features for the train and test sets
python $PWD/grads_supervised.py \
    --seed $SEED \
    --temperature $TEMPERATURE \
    --latent-dim $LATENT_DIM \
    --use-fp16 \
    --model $BACKBONE \
    --batch-size $BATCH_SIZE \
    --embeddings-dim $EMBEDDINGS_DIM \
    --dataset $DATASET \
    --dataset-split train \
    --projection-matrix $PROJECTION_MATRIX \
    --gradients-layers $LAYER_PATH \
    --output-path $OUTPUT_DIR/kl-train.h5

python $PWD/grads_supervised.py \
    --seed $SEED \
    --temperature $TEMPERATURE \
    --latent-dim $LATENT_DIM \
    --use-fp16 \
    --model $BACKBONE \
    --batch-size $BATCH_SIZE \
    --embeddings-dim $EMBEDDINGS_DIM \
    --dataset $DATASET \
    --dataset-split test \
    --projection-matrix $PROJECTION_MATRIX \
    --gradients-layers $LAYER_PATH \
    --output-path $OUTPUT_DIR/kl-test.h5
```

### SimCLR Gradients

The script below extracts the SimCLR gradients for the `banking-77` dataset using a BERT backbone.

```sh
SEED=42
BATCH_SIZE=8
EMBEDDINGS_DIM=768
DATASET=banking-77
BACKBONE=bert-base-uncased
OUTPUT_BASE_DIR=./data
OUTPUT_DIR=$OUTPUT_BASE_DIR/simclr/$DATASET
PROJECTION_MATRIX=bert-projection.pth

# SimCLR-specific parameters
NUM_VIEWS=12
COMPARISON_BATCH_SIZE=256
LATENT_DIM=256

# Layer path and HDF5 dataset name (attn_proj)
LAYER_PATH=backbone.encoder.layer.11.attention.output.dense:attn_proj

# Create a directory to store the feature datasets
mkdir -p $OUTPUT_DIR

python $PWD/grads_simclr.py \
    --seed $SEED \
    --num-views $NUM_VIEWS \
    --comparison-batch-size $COMPARISON_BATCH_SIZE \
    --latent-dim $LATENT_DIM \
    --use-fp16 \
    --model $BACKBONE \
    --batch-size $BATCH_SIZE \
    --embeddings-dim $EMBEDDINGS_DIM \
    --dataset $DATASET \
    --dataset-split train \
    --projection-matrix $PROJECTION_MATRIX \
    --gradients-layers $LAYER_PATH \
    --output-path $OUTPUT_DIR/simclr-train.h5

python $PWD/grads_simclr.py \
    --seed $SEED \
    --num-views $NUM_VIEWS \
    --comparison-batch-size $COMPARISON_BATCH_SIZE \
    --latent-dim $LATENT_DIM \
    --use-fp16 \
    --model $BACKBONE \
    --batch-size $BATCH_SIZE \
    --embeddings-dim $EMBEDDINGS_DIM \
    --dataset $DATASET \
    --dataset-split test \
    --projection-matrix $PROJECTION_MATRIX \
    --gradients-layers $LAYER_PATH \
    --output-path $OUTPUT_DIR/simclr-test.h5
```

### Evaluation

Once you've extracted the features and the KL and SimCLR gradients, run the k-nn evaluation using the command below. This will evaluate the performance of all individual features and all their combinations. This script does not require a GPU!

```sh
PCA_DIM=512
DATASET=banking-77
GRADIENT_DATASET=attn_proj
BACKBONE=bert-base-uncased

python knn.py \
    --normalize \
    --normalize-embeddings \
    --pca-dim $PCA_DIM \
    --train-targets "data/features/$DATASET/train.h5" \
    --test-targets "data/features/$DATASET/test.h5" \
    --train-embeddings-datasets \
    "data/features/$DATASET/train.h5" \
    --test-embeddings-datasets \
    "data/features/$DATASET/test.h5" \
    --train-datasets \
    "data/kl/$DATASET/kl-train.h5:$GRADIENT_DATASET" \
    "data/simclr/$DATASET/simclr-train.h5:$GRADIENT_DATASET" \
    --test-datasets \
    "data/kl/$DATASET/kl-test.h5:$GRADIENT_DATASET" \
    "data/simclr/$DATASET/simclr-test.h5:$GRADIENT_DATASET" \
    --k 20
```

To run the few shot k-nn evaluation use the following command (same as above, with an extra command line argument `--train-size`)

```sh
SHOTS=5
PCA_DIM=512
DATASET=banking-77
GRADIENT_DATASET=attn_proj
BACKBONE=bert-base-uncased

python knn.py \
    --normalize \
    --normalize-embeddings \
    --pca-dim $PCA_DIM \
    --train-size $SHOTS \
    --train-targets "data/features/$DATASET/train.h5" \
    --test-targets "data/features/$DATASET/test.h5" \
    --train-embeddings-datasets \
    "data/features/$DATASET/train.h5" \
    --test-embeddings-datasets \
    "data/features/$DATASET/test.h5" \
    --train-datasets \
    "data/kl/$DATASET/kl-train.h5:$GRADIENT_DATASET" \
    "data/simclr/$DATASET/simclr-train.h5:$GRADIENT_DATASET" \
    --test-datasets \
    "data/kl/$DATASET/kl-test.h5:$GRADIENT_DATASET" \
    "data/simclr/$DATASET/simclr-test.h5:$GRADIENT_DATASET" \
    --k 20
```

Please note that you might get slightly different results compared to the paper, as the projection matrix will be different.
