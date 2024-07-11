# No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations

[Walter Simoncini](https://walter.ashita.nl/)<sup>1</sup>, [Andrei Bursuc](https://abursuc.github.io/)<sup>2</sup>, [Spyros Gidaris](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This repository contains the image classification and retrieval experiments for our paper [No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations](https://fungi.ashita.nl/). Experiments for other modalities and tasks are in their corresponding folders:

- `fungi-text`: k-nearest neighbor text classification using FUNGI obtained from text encoders.
- `fungi-ssast`: k-nearest neighbor audio classification using FUNGI obtained from an SSAST backbone.
- `fungi-hummingbird`: retrieval-based vision in-context learning evaluation on semantic segmentation tasks.

The goal of this repository is to make the experiments illustrated in the paper reproducible, and it's not intented to be used in practice. 

> [!IMPORTANT]  
> If you simply want to extract FUNGI features for your dataset have a look at [fungivision](https://github.com/WalterSimoncini/fungivision), which provides an easy to use and customizable model wrapper for producing FUNGI features.

## Getting Started

This code was developed using `Python 3.10`, so to run the experiments we recommend that you create an appropriate environment and install the required dependencies as follows:

```sh
conda create -n grady python=3.10
conda activate grady

# Install the required Python libraries
pip install -r requirements.txt
```

## Dataset Setup

Most evaluation datasets are loaded via `torchvision`, and require no special setup. In this section, we provide instructions on how to download and generate the ones not available out of the box.

### Stanford Cars

Stanford Cars used to be available in `torchvision`, but links to the original data files are broken. Use the commands below to download and setup the data, and don't forget to manually download `cars_test_annos_withlabels.mat` from [here](https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+%281%29.mat).

```sh
# Based on https://github.com/pytorch/vision/issues/7545 and
# https://github.com/pytorch/vision/issues/7545#issuecomment-1575410733
mkdir /scratch-shared/$USER/cache/cars
kaggle datasets download -p /scratch-shared/$USER/cache/cars jessicali9530/stanford-cars-dataset

# Unpack the dataset and remove recursive structure
cd /scratch-shared/$USER/cache/cars
unzip stanford-cars-dataset.zip

# Move dataset folders to the top level
mv cars_test/ cars_test_tmp
mv cars_test_tmp/cars_test/ .
rm -rf cars_test_tmp

mv cars_train/ cars_train_tmp
mv cars_train_tmp/cars_train/ .
rm -rf cars_train_tmp

wget https://github.com/pytorch/vision/files/11644847/car_devkit.tgz

# Download the test annotations manually from this URL
# https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+%281%29.mat

# Go back to the home folder
cd $HOME

# Wrap the cars dataset into the proper folder structure, i.e. cars/stanford_cars/...
mv /scratch-shared/$USER/cache/cars /scratch-shared/$USER/cache/stanford_cars
mkdir /scratch-shared/$USER/cache/cars
mv /scratch-shared/$USER/cache/stanford_cars /scratch-shared/$USER/cache/cars

# Unpack the devkit
cd /scratch-shared/$USER/cache/cars/stanford_cars/
tar -xf car_devkit.tgz
```

### Oxford Pets

> [!WARNING]
> The pets dataset is no longer available from the original source. If the issue is only temporary you will be able to use the following commands to download and build its corresponding HuggingFace dataset.

```sh
mkdir /scratch-shared/$USER/cache/pets

# Download the images and annotations
wget -P /scratch-shared/$USER/cache/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget -P /scratch-shared/$USER/cache/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

# Unpack the images and annotations
cd /scratch-shared/$USER/cache/pets

tar -xf images.tar.gz
tar -xf annotations.tar.gz

# Go back to the home folder (or the folder containing this repository)
cd $HOME/knn

# Generate an HuggingFace dataset version of Pets
sh jobs/datasets/generate_pets.job
```

### ImageNet 100

Download ImageNet-100 from Kaggle and build its corresponding HuggingFace dataset.

```sh
kaggle datasets download -p /scratch-shared/$USER/cache ambityga/imagenet100

# Unpack and structure the dataset
cd /scratch-shared/$USER/cache
mkdir imagenet100 & mv imagenet100.zip imagenet100
cd imagenet100
unzip imagenet100.zip

# Update the script arguments as needed before scheduling this job
sh jobs/datasets/generate_imagenet100.job
```

### CUB 200 2011

Download CUB 200 (2011) and build its corresponding HuggingFace dataset.

```sh
WORKING_DIR=$PWD
DATASET_DIR=/scratch-shared/$USER/cache/cub

mkdir $DATASET_DIR
wget -P $DATASET_DIR https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz

cd $DATASET_DIR
tar -xf CUB_200_2011.tgz
cd $WORKING_DIR

sh jobs/datasets/generate_cub.job
```

### Oxford and Paris Retrieval

Download the Oxford and Paris landmarks datasets from Kaggle. These datasets are only used for the retrieval experiments.

```sh
BASE_DIR=/scratch-shared/$USER/cache/retrieval-landmarks

mkdir -p $BASE_DIR
kaggle datasets download -p $BASE_DIR qiubit/roxfordparis
cd $BASE_DIR

unzip roxfordparis.zip
```

## Reproducing Experiments

Most of the experiments in the paper can be reproduced using the scripts in the `jobs` directory. The scripts assume you have a scratch folder whose path is `/scratch-shared/$USER`, make sure to change it as appropriate.

### Image Classification Experiments

#### Computing Gradients and Embeddings

To run the image classification experiments you first have to extract the embeddings and gradient features for both the training and test splits of a dataset. To extract the model embeddings use the `jobs/extract.job` script, and customize the `DATASETS` and `MODEL_TYPES` arrays as appropriate. The full list of supported models and datasets is available in `src/enums.py`.

Once you've extracted the embeddings you can then extract the gradient features for each individual loss/objective. To do so use one of the `jobs/grads_xxx_multi.job` scripts, where `xxx` is the objective name. You can customize the parameters to be used for the objective, the datasets and backbones in the scripts.

For each backbones, in addition to its name you must also specify the linear layer from which gradients should be extracted from and the corresponding dataset name in the output `HDF5` gradients file in the `LAYER_PATHS` array. Each entry should be in the format `path.to.layer:out_dataset_name`, e.g. to extract the gradients for the last attention output projection and save them in the `attn_proj` dataset you should use `backbone.blocks.11.attn.proj:attn_proj`. The word `dataset` in this context refers to one of the tensor arrays stored in each HDF5 file, as they are indexed by name.

Most models from `timm` follow the `backbone.blocks.xx.attn.proj` structure, while `torchvision` models follow the `backbone.encoder.layers.encoder_layer_xx.self_attention.out_proj` structure, where `xx` is the transformer block number.

Finally, you will also have to specify the path to a random projection matrix in the `PROJECTION_MATRICES` array, which is used to downsample the gradients to a smaller vector, so that they are manageable with respect to storage and compute. You can generate one using the following command

```sh
python generate_projection_matrix.py \
    --projection-dim $EMBEDDINGS_DIM \
    --gradients-dim $GRADIENTS_DIM \
    --output-path /scratch-shared/$USER/path_to_matrix.pth
```

Where `$EMBEDDINGS_DIM` is the dimension of the projected gradients and `$GRADIENTS_DIM` is the dimension of the gradients matrix. Below we list these dimensions as `($EMBEDDINGS_DIM, $GRADIENTS_DIM)` tuples for some models, assuming that gradients are always extracted from the attention output projection.

- ViT-S/16: (384, 147840)
- ViT-B/16: (768, 590592)
- ViT-L/16: (1024, 1049600)

> [!WARNING]
> Make sure you're using the same projection to extract gradients for the same dataset but from different objectives! Using different projections may lead to significant performance degradation.

#### Evaluating FUNGI Features

Once you've extracted gradient features and embeddings you can evaluate their performance in k-nearest neighbor classification using the `jobs/concat.job` (or `jobs/concat_few_shot.job`). You can choose the PCA dimensionality (set it to a large number to effectively disable it), and the datasets and backbones to evaluate. Don't forget to update the data paths as appropriate!

The script will evaluate the performance of all possible combinations of gradients and embeddings. You can run the same experiments for linear classification using `jobs/concat_linear.job` (or the few shot counterpart), but only incremental combinations will be evaluated (i.e. embeddings, embeddings + kl, ...). To disable this behavior and evaluate every combination comment out line `172` of the `train_linear_concat_grads.py` script.

### Retrieval Experiments

You can replicate the retrieval experiments by following the same steps for image classification, but instead using the scripts in the `jobs/retrieval` folder.

## Tests

We provide some automatic tests used to validate the gradients extraction and whether the manually downloaded datasets were processed correctly. You can run them using the following commands:

```sh
mkdir -p cache/models
pytest
```

The gradient sanity check tests should be run on CPU, as running them on a GPU may produce slighly different gradients depending on the batch size and cause the test to fail.
