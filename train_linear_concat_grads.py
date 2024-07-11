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

from sklearn import metrics
from sklearn.decomposition import PCA
from cyanure.estimators import Classifier
from src.utils.seed import seed_everything
from src.utils.sampling import sample_balanced
from src.utils.logging import configure_logging
from src.utils.metrics import mean_per_class_accuracy
from sklearn.model_selection import StratifiedShuffleSplit


def main(args):
    assert len(args.train_datasets) == len(args.test_datasets), "there should be an equal number of training and testing datasets (gradients)"
    assert len(args.train_datasets) == len(args.valid_datasets), "there should be an equal number of training and validation datasets (gradients)"

    assert len(args.train_embeddings_datasets) == len(args.test_embeddings_datasets), "there should be an equal number of training and testing datasets (embeddings)"
    assert len(args.train_embeddings_datasets) == len(args.valid_embeddings_datasets), "there should be an equal number of training and validation datasets (embeddings)"

    logging.info(f"the training dataset size is {args.train_size}")

    # Read the training and testing targets
    test_targets = h5py.File(args.test_targets)["targets"][:]
    train_targets = h5py.File(args.train_targets)["targets"][:]

    # Check that the validation dataset exists, and if that's not the case
    # create one via a 80/20 stratified split of the training set
    validation_exists = os.path.exists(args.valid_targets)

    if not validation_exists:
        logging.warning(f"the validation set targets were not found! The validation set will be created with an 80/20 split from the training set")

        train_val_split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)

        train_indices, val_indices = list(train_val_split.split(train_targets, train_targets))[0]
        train_indices, val_indices = np.sort(train_indices), np.sort(val_indices)

        valid_targets = train_targets[val_indices]
        train_targets = train_targets[train_indices]
    else:
        valid_targets = h5py.File(args.valid_targets)["targets"][:]

    logging.info(f"the script arguments were {args}")
    logging.info(f"the train embeddings datasets are {args.train_embeddings_datasets}")
    logging.info(f"the test embeddings datasets are {args.test_embeddings_datasets}")
    logging.info(f"the validation embeddings datasets are {args.valid_embeddings_datasets}")

    # Split datasets into filename and data key
    args.train_datasets = [ds.split(":") for ds in args.train_datasets]
    args.test_datasets = [ds.split(":") for ds in args.test_datasets]
    args.valid_datasets = [ds.split(":") for ds in args.valid_datasets]

    # few-shot evaluation, i.e. select the whole training
    # set or only a subset to fit the k-NN model
    if args.train_size is None:
        # Select the whole dataset
        shots_training_indices = np.arange(len(train_targets))
    else:
        shots_training_indices = sample_balanced(
            targets=train_targets,
            n_examples_per_class=args.train_size,
            shuffle=False
        )

        shots_training_indices = np.sort(shots_training_indices)

    # We initialize the lists of features with the embeddings, as we do
    # not project them since they are relatively low-dimensional with
    # respect to the gradients
    test_features = [
        h5py.File(filename)["embeddings"][:] for filename in args.test_embeddings_datasets
    ]

    train_features = [
        h5py.File(filename)["embeddings"][:] for filename in args.train_embeddings_datasets
    ]

    if not validation_exists:
        valid_features = [x[val_indices] for x in train_features]
        train_features = [x[train_indices][shots_training_indices] for x in train_features]
    else:
        train_features = [x[shots_training_indices] for x in train_features]
        valid_features = [
            h5py.File(filename)["embeddings"][:] for filename in args.valid_embeddings_datasets
        ]

    logging.info(f"loaded {len(test_features)} embedding datasets")

    if args.normalize_embeddings:
        logging.info(f"normalizing embeddings...")

        for i in range(len(train_features)):
            train_features[i] = F.normalize(torch.tensor(train_features[i]), dim=-1, p=2).numpy()
            test_features[i] = F.normalize(torch.tensor(test_features[i]), dim=-1, p=2).numpy()
            valid_features[i] = F.normalize(torch.tensor(valid_features[i]), dim=-1, p=2).numpy()

    logging.info("projecting the gradients datasets")

    # We assume training and testing datasets are paired (i.e. given in the
    # same arguments order). This is necessary as we must have a common
    # projection matrix for the training and testing data
    for train_meta, test_meta, valid_meta in zip(args.train_datasets, args.test_datasets, args.valid_datasets):
        # Open the datasets for training and testing
        test_file_path, test_data_key = test_meta
        train_file_path, train_data_key = train_meta
        valid_file_path, valid_data_key = valid_meta

        test_dataset = h5py.File(test_file_path)[test_data_key][:]

        # While in principle not necessary (e.g. for a few-shot evaluation) we
        # read the whole training dataset into memory as sequential reads are
        # infinitely faster than randomly accessing indices for large datasets,
        # which happens when doing a train/val split
        train_dataset = h5py.File(train_file_path)[train_data_key][:]
        train_dataset = train_dataset[:]

        if not validation_exists:
            valid_dataset = train_dataset[val_indices]
            train_dataset = train_dataset[train_indices][shots_training_indices]
        else:
            train_dataset = train_dataset[shots_training_indices]
            valid_dataset = h5py.File(valid_file_path)[valid_data_key][:]

        if args.normalize:
            test_dataset = F.normalize(torch.tensor(test_dataset), dim=-1, p=2).numpy()
            train_dataset = F.normalize(torch.tensor(train_dataset), dim=-1, p=2).numpy()
            valid_dataset = F.normalize(torch.tensor(valid_dataset), dim=-1, p=2).numpy()

        test_features.append(test_dataset)
        train_features.append(train_dataset)
        valid_features.append(valid_dataset)

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

    for n in range(1, len(train_features) + 1):
        logging.info(f"testing length {n} combinations")

        # only select the following combinations:
        # train
        # train + kl
        # train + kl + dino
        # train + kl + dino + simclr
        valid_combinations = list(itertools.combinations(np.arange(len(train_features)), n))
        valid_combinations = [valid_combinations[0]]

        for combination in valid_combinations:
            human_readable_combination = [training_dataset_names[i] for i in combination]
            human_readable_combination = ";".join(human_readable_combination)

            logging.info(f"testing datasets combination {human_readable_combination.replace(';', ' ; ')}")
            logging.info(f"the dataset indices are {combination}")

            step_test_features = np.concatenate([test_features[i] for i in combination], axis=-1)
            step_train_features = np.concatenate([train_features[i] for i in combination], axis=-1)
            step_valid_features = np.concatenate([valid_features[i] for i in combination], axis=-1)

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
                step_valid_features = pca.transform(step_valid_features)
            else:
                if args.pca_dim > step_train_features.shape[0]:
                    logging.warning(f"not running PCA as the dataset is smaller than the projection dimensionality {step_train_features.shape[0]} < {args.pca_dim}")
                else:
                    logging.warning(f"not running PCA as the PCA dimension is smaller than the features {step_train_features.shape[1]} <= {args.pca_dim}")

            logging.info(f"the shapes of training and test features are: {(step_train_features.shape, step_test_features.shape)}")
            logging.info(f"the validation set has shape: {step_valid_features.shape}")
            logging.info(f"selecting the best L2 lambda on the validation set...")

            logging.info(f"training the sweep classifiers for {args.max_opt_iter} epochs")
            logging.info(f"training the final classifier for {args.max_iter} epochs")

            # Convert the datasets to double to avoid numerical errors
            step_test_features = step_test_features.astype(np.double)
            step_train_features = step_train_features.astype(np.double)
            step_valid_features = step_valid_features.astype(np.double)

            # Regularization hyperparameter search
            lambda_values = np.linspace(0.000005, 0.0005, 5)
            lambda_val_accuracies = np.zeros(lambda_values.shape[0])

            for i, reg_lambda in enumerate(lambda_values):
                classifier = build_classifier(reg_lambda=reg_lambda, max_iter=args.max_opt_iter)
                classifier.fit(step_train_features, train_targets[shots_training_indices])

                train_accuracy = classifier.score(step_train_features, train_targets[shots_training_indices])
                valid_accuracy = classifier.score(step_valid_features, valid_targets)

                lambda_val_accuracies[i] = valid_accuracy

                logging.info(f"lambda {reg_lambda} -- train accuracy: {train_accuracy}, valid accuracy: {valid_accuracy}")

            selected_lambda = lambda_values[lambda_val_accuracies.argmax()]

            logging.info(f"fitting a logistic classifier with lambda {selected_lambda}")

            classifier = build_classifier(reg_lambda=selected_lambda, max_iter=args.max_iter)

            if not validation_exists and args.train_size is None:
                # If the validation was obtained as an 80/20 split of the training set
                # re-create the training set and use it to train the final classifier.
                # This is only applied in the full dataset evaluation, as for the few
                # shots one the training set already has an appropriate size
                step_train_features = np.concatenate([step_train_features, step_valid_features], axis=0).astype(np.double)
                step_train_targets = np.concatenate([train_targets, valid_targets], axis=0).astype(np.double)

                classifier.fit(step_train_features, step_train_targets)
            else:
                classifier.fit(step_train_features, train_targets[shots_training_indices])

            logging.info("testing the logistic classifier")

            test_predictions = classifier.predict(step_test_features)

            correct_predictions = (test_predictions == test_targets[:]).sum()
            accuracy = correct_predictions / len(test_targets)

            logging.info(f"the test accuracy was {accuracy}")

            mean_per_class_acc = mean_per_class_accuracy(
                preds=test_predictions,
                targets=test_targets[:]
            )

            logging.info(f"the mean per-class accuracy was {mean_per_class_acc}")
            logging.info(f"-" * 100)

            if len(np.unique(train_targets[shots_training_indices])) == 2:
                # We also report the AUC for datasets with binary labels.
                # We assume the positive class is the one with label 1
                positive_class_probs = classifier.predict_proba(step_test_features)[:, 1]
                fpr, tpr, _ = metrics.roc_curve(test_targets[:], positive_class_probs, pos_label=1)

                auc = metrics.auc(fpr, tpr)

                logging.info(f"the auc was {auc}")
            else:
                auc = None

            performance_log[human_readable_combination] = {
                "accuracy": [accuracy],
                "mean-per-class-accuracy": [mean_per_class_acc],
                "auc": [auc],
                "predictions": [test_predictions.tolist()],
                "validation_accuracies": [lambda_val_accuracies.tolist()],
                "validation_lambdas": [lambda_values.tolist()]
            }

    if args.logs_output_path:
        with open(args.logs_output_path, "w") as logs_file:
            logs_file.write(json.dumps({
                "args": vars(args),
                "metrics": performance_log,
                "test_targets": test_targets[:].tolist()
            }))

        logging.info(f"saved logs to {args.logs_output_path}")


def build_classifier(reg_lambda: float, max_iter: int = 300) -> Classifier:
    return Classifier(
        loss="multiclass-logistic",
        penalty="l2",
        fit_intercept=False,
        lambda_1=reg_lambda,
        n_threads=-1,
        tol=1e-3,
        solver="auto",
        random_state=args.seed,
        max_iter=max_iter,
        verbose=False
    )


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Trains and test a k-NN estimator by concatenating gradients from different model/losses (and embeddings)")

    # Seed (controls the random projections and k-NN seed)
    parser.add_argument("--seed", type=int, default=42)

    # Data arguments
    parser.add_argument("--train-targets", type=str, required=True, help="The training targets. Must be an HDF5 file with a `targets` key")
    parser.add_argument("--valid-targets", type=str, required=True, help="The validation targets. Must be an HDF5 file with a `targets` key")
    parser.add_argument("--test-targets", type=str, required=True, help="The test targets. Must be an HDF5 file with a `targets` key")

    parser.add_argument("--train-datasets", type=str, nargs="*", required=False, default=[], help="The training datasets. Expressed as filename:data_key")
    parser.add_argument("--valid-datasets", type=str, nargs="*", required=False, default=[], help="The validation datasets. Expressed as filename:data_key")
    parser.add_argument("--test-datasets", type=str, nargs="*", required=False, default=[], help="The test datasets. Expressed as filename:data_key")

    parser.add_argument("--train-embeddings-datasets", type=str, nargs="*", required=False, default=[], help="The training embeddings dataset filenames")
    parser.add_argument("--valid-embeddings-datasets", type=str, nargs="*", required=False, default=[], help="The validation embeddings dataset filenames")
    parser.add_argument("--test-embeddings-datasets", type=str, nargs="*", required=False, default=[], help="The test embeddings dataset filenames")

    # Data preprocessing
    parser.add_argument("--pca-dim", type=int, default=768, help="The output dimensionality of the PCA model")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False, help="Whether to normalize the projected gradient vectors")
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=False, help="Whether to normalize the embeddings")

    # Model parameters
    parser.add_argument("--max-iter", type=int, required=False, default=300, help="Maximum training iterations")
    parser.add_argument("--max-opt-iter", type=int, required=False, default=300, help="Maximum training iterations for the L2 regularization parameter search")

    # Few show evaluation
    parser.add_argument("--train-size", type=int, required=False, default=None, help="Percentage of the training set used to fit the k-NN model. If > 1 it's considered to be the absolute number of samples per class.")

    # Logging
    parser.add_argument("--logs-output-path", type=str, required=False, default=None, help="Where to save the metrics logs (if not specified no log will be saved)")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
