import os
import scann
import torch
import argparse

from src.hbird_eval import create_memory_bank
from src.utils.misc import seed_everything, model_feature_dim


def main(args):
    print(f"the arguments are {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)

    embedding_size = model_feature_dim(
        model=model,
        device=device,
        image_size=args.input_size
    )

    def token_features(model, imgs):
        with torch.no_grad():
            patches = model.get_intermediate_layers(imgs)[0][:, 1:]

        return patches, None

    create_memory_bank(
        model.to(device),
        d_model=embedding_size,
        patch_size=args.patch_size,
        batch_size = args.batch_size,
        input_size=args.input_size,
        # How many iterations of augmentations to use on top of the training dataset in order to generate the memory
        augmentation_epoch=args.augmentation_epochs,
        device=device,
        # Function that maps an image to a set of patches
        ftr_extr_fn=token_features,
        dataset_name="voc",
        data_dir=args.data_dir,
        memory_size=args.memory_size,
        f_mem_p=args.memory_bank,
        l_mem_p=args.memory_bank_labels,
        num_train_samples=args.num_train_samples
    )

    print(f"saved the memory bank to {args.memory_bank}")
    print(f"saved the memory bank labels to {args.memory_bank_labels}")
    print("creating a scann index...")

    memory = torch.load(args.memory_bank)

    voc_index = scann.scann_ops_pybind.builder(
        memory,
        args.num_neighbors,
        "dot_product"
    ).tree(
        num_leaves=args.num_leaves,
        num_leaves_to_search=args.num_leaves_to_search,
        training_sample_size=memory.shape[0]
    ).score_ah(
        dimensions_per_block=args.dimensions_per_block,
        anisotropic_quantization_threshold=args.anisotropic_quantization_threshold
    ).reorder(args.num_reordering_candidates).build()

    # Make sure the index directory exists
    os.makedirs(args.scann_index, exist_ok=True)

    voc_index.serialize(args.scann_index)

    print(f"saved the scann index to {args.scann_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the model patches")
    parser.add_argument("--model", type=str, required=True, help="Model name (dino_xxx)")
    parser.add_argument("--memory-size", type=int, default=None, help="The size of the memory bank")
    parser.add_argument("--augmentation-epochs", type=int, default=1, help="The number of augmentation epochs")

    # NN index parameters
    parser.add_argument("--num-neighbors", type=int, default=30, help="The number of neighbors retrieved from the NN index")
    parser.add_argument("--num-leaves", type=int, default=512, help="The number of leaves for the NN index")
    parser.add_argument("--num-leaves-to-search", type=int, default=32, help="The number of leaves to search for the NN index")
    parser.add_argument("--num-reordering-candidates", type=int, default=120, help="The number of candidates to rerank for the NN index")
    parser.add_argument("--anisotropic-quantization-threshold", type=float, default=0.2, help="The anisotropic quantization threshold for the NN index")
    parser.add_argument("--dimensions-per-block", type=int, default=4, help="The dimensions per block for the NN index")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="VOCSegmentation", help="Path to the VOC dataset")
    parser.add_argument("--memory-bank", type=str, required=True, help="Where to save the memory bank")
    parser.add_argument("--memory-bank-labels", type=str, required=True, help="Where to save the memory bank labels")
    parser.add_argument("--scann-index", type=str, required=True, help="Where to save the memory bank index")
    parser.add_argument("--num-train-samples", type=int, required=False, default=None, help="How many dataset samples should be used? None means all samples will be used")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
