"""
    This script concatenates the (projected) gradients
    from multiple gradient datasets and uses them to
    fit a k-NN classifier.
"""
import os
import h5py
import json
import torch
import logging
import argparse
import itertools
import numpy as np
import torch.nn.functional as F

from src.utils.metrics import compute_map
from sklearn.decomposition import PCA
from src.utils.seed import seed_everything
from src.utils.logging import configure_logging

from src.datasets import load_dataset
from src.enums import DatasetType, DatasetSplit


def main(args):
    assert len(args.train_datasets) == len(args.test_datasets), "there should be an equal number of training and testing datasets (gradients)"
    assert len(args.train_embeddings_datasets) == len(args.test_embeddings_datasets), "there should be an equal number of training and testing datasets (embeddings)"

    # Read the training and testing targets
    test_targets = h5py.File(args.test_targets)["targets"][:]

    logging.info(f"the script arguments were {args}")
    logging.info(f"the train embeddings datasets are {args.train_embeddings_datasets}")
    logging.info(f"the test embeddings datasets are {args.test_embeddings_datasets}")

    # Split datasets into filename and data key
    args.train_datasets = [ds.split(":") for ds in args.train_datasets]
    args.test_datasets = [ds.split(":") for ds in args.test_datasets]

    # We initialize the lists of features with the embeddings, as we do
    # not project them since they are relatively low-dimensional with
    # respect to the gradients
    test_features = [
        h5py.File(filename)["embeddings"][:] for filename in args.test_embeddings_datasets
    ]

    train_features = [
        h5py.File(filename)["embeddings"][:] for filename in args.train_embeddings_datasets
    ]

    logging.info(f"loaded {len(test_features)} embedding datasets")

    if args.normalize_embeddings:
        logging.info(f"normalizing embeddings...")

        for i in range(len(train_features)):
            train_features[i] = F.normalize(torch.tensor(train_features[i]), dim=-1, p=2).numpy()
            test_features[i] = F.normalize(torch.tensor(test_features[i]), dim=-1, p=2).numpy()

    logging.info("projecting the gradients datasets")

    # We assume training and testing datasets are paired (i.e. given in the
    # same arguments order). This is necessary as we must have a common
    # projection matrix for the training and testing data
    for train_meta, test_meta in zip(args.train_datasets, args.test_datasets):
        # Open the datasets for training and testing
        test_file_path, test_data_key = test_meta
        train_file_path, train_data_key = train_meta

        test_dataset = h5py.File(test_file_path)[test_data_key][:]
        train_dataset = h5py.File(train_file_path)[train_data_key][:]

        if args.normalize:
            test_dataset = F.normalize(torch.tensor(test_dataset), dim=-1, p=2).numpy()
            train_dataset = F.normalize(torch.tensor(train_dataset), dim=-1, p=2).numpy()

        test_features.append(test_dataset)
        train_features.append(train_dataset)

    # We have now projected all the input datasets, and the next step is testing
    # all possible embeddings/gradients combinations of any size (1 to N). So:
    #
    # 1. Create an array of dataset names, so that we can log which set of
    #    datasets led to which metrics
    # 2. For every n in [1, N] (N = number of datasets) generate all possible
    #    combinations of size n and test their performance

    # Create an array of dataset names, used to log a human-readable
    # list of the data used to fit the k-NN model
    training_dataset_names = [os.path.split(x)[1] for x in args.train_embeddings_datasets]
    training_dataset_names += [
        f"{os.path.split(path)[1]}:{dataset_name}" for (path, dataset_name) in args.train_datasets
    ]

    performance_log = {}

    original_train_dataset = load_dataset(
        type_=args.dataset_name,
        split=DatasetSplit.TRAIN,
        cache_dir=args.cache_dir
    )

    for n in range(1, len(train_features) + 1):
        logging.info(f"testing length {n} combinations")

        for combination in itertools.combinations(np.arange(len(train_features)), n):
            human_readable_combination = [training_dataset_names[i] for i in combination]
            human_readable_combination = ";".join(human_readable_combination)

            logging.info(f"testing datasets combination {human_readable_combination.replace(';', ' ; ')}")
            logging.info(f"the dataset indices are {combination}")

            step_test_features = np.concatenate([test_features[i] for i in combination], axis=-1)
            step_train_features = np.concatenate([train_features[i] for i in combination], axis=-1)

            # Reduce all combinations to the same dimensionality
            # via PCA for a fair evaluation. If the PCA dimensionality
            # is the same as the embeddings they won't be compressed,
            # but combinations will.
            if args.pca_dim < step_train_features.shape[0] and args.pca_dim < step_train_features.shape[1]:
                # We can only run PCA if the output dimensionality is smaller than the dataset size
                logging.info(f"testing PCA dimensionality of {args.pca_dim}")

                pca = PCA(n_components=args.pca_dim, svd_solver="full", random_state=args.seed)
                pca.fit(step_train_features)

                step_test_features = pca.transform(step_test_features)
                step_train_features = pca.transform(step_train_features)
            else:
                if args.pca_dim > step_train_features.shape[0]:
                    logging.warning(f"not running PCA as the dataset is smaller than the projection dimensionality {step_train_features.shape[0]} < {args.pca_dim}")
                else:
                    logging.warning(f"not running PCA as the PCA dimension is smaller than the features {step_train_features.shape[1]} <= {args.pca_dim}")

            logging.info(f"the shapes of training and test features are: {(step_train_features.shape, step_test_features.shape)}")
            logging.info(f"evaluating retrieval...")

            # Code taken from https://github.com/facebookresearch/dino/blob/main/eval_image_retrieval.py
            step_test_features = torch.tensor(step_test_features)
            step_train_features = torch.tensor(step_train_features)

            ############################################################################
            # Step 2: similarity
            sim = torch.mm(step_train_features, step_test_features.T)
            ranks = torch.argsort(-sim, dim=0).cpu().numpy()

            ############################################################################
            # Step 3: evaluate
            gnd = original_train_dataset.cfg['gnd']
            # evaluate ranks
            ks = [1, 5, 10]
            # search for easy & hard
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk']])
                gnd_t.append(g)
            mapM, _, mprM, _ = compute_map(ranks, gnd_t, ks)
            # search for hard
            gnd_t = []
            for i in range(len(gnd)):
                g = {}
                g['ok'] = np.concatenate([gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
                gnd_t.append(g)

            mapH, _, mprH, _ = compute_map(ranks, gnd_t, ks)

            logging.info('mAP M: {}, H: {}'.format(mapM, mapH))
            logging.info('mP@k{} M: {}, H: {}'.format(np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

            performance_log[human_readable_combination] = {
                "mAP-M": [mapM],
                "mAP-H": [mapH],
                "mP@k": ks,
                "mP@k-M": mprM.tolist(),
                "mP@k-H": mprH.tolist()
            }

    if args.logs_output_path:
        # Enums must be stringified to serialize them to json
        args.dataset_name = str(args.dataset_name)

        with open(args.logs_output_path, "w") as logs_file:
            logs_file.write(json.dumps({
                "args": vars(args),
                "metrics": performance_log,
                "test_targets": test_targets[:].tolist()
            }))

        logging.info(f"saved logs to {args.logs_output_path}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Trains and test a k-NN estimator by concatenating gradients from different model/losses (and embeddings)")

    # Seed (controls the random projections and k-NN seed)
    parser.add_argument("--seed", type=int, default=42)

    # Data arguments
    parser.add_argument("--test-targets", type=str, required=True, help="The test targets. Must be an HDF5 file with a `targets` key")

    # Dataset arguments, for the original ground truth
    parser.add_argument("--dataset-name", type=DatasetType, choices=[DatasetType.OXFORD_LANDMARKS, DatasetType.PARIS_LANDMARKS], required=True, help="The dataset being tested")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    parser.add_argument("--train-datasets", type=str, nargs="*", required=False, default=[], help="The training datasets. Expressed as filename:data_key")
    parser.add_argument("--test-datasets", type=str, nargs="*", required=False, default=[], help="The test datasets. Expressed as filename:data_key")

    parser.add_argument("--train-embeddings-datasets", type=str, nargs="*", required=False, default=[], help="The training embeddings dataset filenames")
    parser.add_argument("--test-embeddings-datasets", type=str, nargs="*", required=False, default=[], help="The test embeddings dataset filenames")

    # Data preprocessing
    parser.add_argument("--pca-dim", type=int, default=768, help="The output dimensionality of the PCA model")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False, help="Whether to normalize the projected gradient vectors")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=False, help="Whether to normalize the embeddings")

    # Logging
    parser.add_argument("--logs-output-path", type=str, required=False, default=None, help="Where to save the metrics logs (if not specified no log will be saved)")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
