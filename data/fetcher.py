import torch
from munch import Munch


class Fetcher:
    def __init__(self, loader, args):
        self.loader = loader
        self.device = torch.device(args.device)
        self.latent_dim = args.latent_dim

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        inputs = Munch(x=x, y=y)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
