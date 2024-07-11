import os
import torch
import random
import numpy as np


def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)

	os.environ["PYTHONHASHSEED"] = str(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
