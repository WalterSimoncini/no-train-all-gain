# FUNGI SSAST: Gradients Extraction from Audio Backbones

[Walter Simoncini](https://walter.ashita.nl/)<sup>1</sup>, [Andrei Bursuc](https://abursuc.github.io/)<sup>2</sup>, [Spyros Gidaris](https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en)<sup>2</sup>, [Yuki M. Asano](https://yukimasano.github.io/)<sup>1</sup>.

1. [QUVA Lab](https://ivi.fnwi.uva.nl/quva/), University of Amsterdam.
2. [valeo.ai](https://www.valeo.com/en/valeo-ai/), Paris, France.

This repository contains the audio classification experiments for our paper [No Train, all Gain: Self-Supervised Gradients Improve Deep Frozen Representations](https://fungi.ashita.nl/). This repository is built on top of [SSAST](https://github.com/YuanGongND/ssast), and implements the extraction of KL and SimCLR gradients for an audio spectrogram transformer. In particular, we use the [Patch SSAST](https://ojs.aaai.org/index.php/AAAI/article/view/21315) backbone, and implement the gradients extraction for two datasets:

- [ESC 50](https://github.com/karolpiczak/ESC-50)
- [SpeechCommands V2](https://arxiv.org/pdf/1804.03209)

We use the SSAST `AudioDataset` implementation to load and process the data, and use the default parameters for each dataset.

## Setup

Create a conda environment and install the required dependencies using the commands below:

```sh
conda create -n ssast python=3.10
pip install -r requirements.txt
```

### Downloading and Preprocessing Datasets

Before preprocessing the datasets make sure `sox` is installed on your machine.

#### ESC 50

Use the script below to download and preprocess the ESC 50 dataset. Replace the value of `DESTINATION_DIR` as appropriate. This script assumes you're configuring the dataset on a cluster with a `/scratch-shared` folder.

```sh
cd src/prep_data/esc50/

export DESTINATION_DIR="/scratch-shared/$USER/cache/esc-50"
export SOURCE_DIR="$(pwd -P)/data"

mkdir -p data
mkdir -p $DESTINATION_DIR

python prep_esc50.py

sed -i "s+$SOURCE_DIR+$DESTINATION_DIR+g" data/datafiles/esc_eval_data_1.json
sed -i "s+$SOURCE_DIR+$DESTINATION_DIR+g" data/datafiles/esc_train_data_1.json

mv data/* $DESTINATION_DIR
cp esc_class_labels_indices.csv $DESTINATION_DIR
```

#### SpeechCommands V2

Use the script below to download and preprocess SpeechCommands V2. Replace `DESTINATION_DIR` as appropriate.

```sh
cd src/prep_data/speechcommands/

export DESTINATION_DIR="/scratch-shared/$USER/cache/speechcommands"
export SOURCE_DIR="$(pwd -P)/data"

mkdir -p data
mkdir -p $DESTINATION_DIR

python prep_sc.py

sed -i "s+$SOURCE_DIR+$DESTINATION_DIR+g" data/datafiles/speechcommand_eval_data.json
sed -i "s+$SOURCE_DIR+$DESTINATION_DIR+g" data/datafiles/speechcommand_train_data.json
sed -i "s+$SOURCE_DIR+$DESTINATION_DIR+g" data/datafiles/speechcommand_valid_data.json

mv data/* $DESTINATION_DIR
cp speechcommands_class_labels_indices.csv $DESTINATION_DIR
```

## Running Experiments

Once you've set up the environment and downloaded the datasets, you can start extracting gradients. But before then, generate a projection matrix using the following command, as the downsampling of gradients is done on the fly as they are being extracted

```sh
python generate_projection.py \
    --projection-dim 768 \
    --gradients-dim 590592 \
    --output-path ssast-projection.pth
```

You can then run the scripts in the `jobs` folder to extract features, kl and simclr gradients. Update the script parameters as needed, especially with regards to the datasets cache and output directories. You may also need to tweak the batch size, depending on your available VRAM.
