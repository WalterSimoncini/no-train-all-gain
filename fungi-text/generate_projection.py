import torch
import logging
import argparse

from src.utils.misc import seed_everything, configure_logging
from src.utils.compression import (
    suggested_scaling_factor,
    generate_projection_matrix
)


def main(args):
    logging.info(f"creating a projection matrix of size {(args.projection_dim, args.gradients_dim)}")

    scaling = suggested_scaling_factor(projection_dim=args.projection_dim)
    projection = generate_projection_matrix(
        dims=(args.projection_dim, args.gradients_dim),
        device=torch.device("cpu")
    )

    torch.save({
        "scaling": scaling,
        "projection": projection
    }, args.output_path)

    logging.info(f"saved the projection matrix to {args.output_path}")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Generates a random projection matrix and saves it to disk")

    parser.add_argument("--seed", default=42, type=int, help="The seed for random number generators")
    parser.add_argument("--output-path", type=str, help="Where to save the projection matrix")

    parser.add_argument("--projection-dim", type=int, required=True, help="The output dimensionality of projected gradients")
    parser.add_argument("--gradients-dim", type=int, required=True, help="The original gradients dimensionality")    

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
