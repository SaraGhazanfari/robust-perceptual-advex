import torch


class StaticVars:
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('mps')
