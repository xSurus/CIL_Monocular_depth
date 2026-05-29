import random

import numpy as np
import torch


def get_seed(cfg, default=42):
    return int(cfg.get('seed', default))


def seed_everything(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
