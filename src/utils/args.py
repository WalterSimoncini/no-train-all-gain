from typing import List, Dict


def parse_gradient_targets(targets: List[str]) -> Dict[str, str]:
    """
        Parses the targets for gradient extraction, which are
        specified in the command line argument as an array of
        items in the format layer_path:out_dataset and returns a
        mapping from the layer_path to the dataset
    """
    return dict([
        # Each layer is specified as layer_path:dataset_name,
        # so to get the keys and values we can simply split
        # the string by the ":" character
        target.split(":") for target in targets
    ])
