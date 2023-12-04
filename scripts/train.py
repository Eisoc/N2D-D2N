"""

"""
import os
import datetime
import random
import json
import argparse
from pprint import pprint

import torch
import torch.optim as optim

import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

from models.model import Net
from utils.dataset import make_data_loader
from utils.trainer import Trainer

torch.autograd.set_detect_anomaly(True)


def main(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Use following parameters:")
    for k, v in vars(args).items():
        print("{}\t{}".format(k, v))
    print("=======================================\n")
    pprint(config)
    print("=======================================\n")

    train_loader, val_loader = make_data_loader(config["dataloader"], config["train"]["stages"])

    model = Net(config["model"])
    model.to("cuda")
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = ExponentialLR(optimizer, gamma=config["optimizer"]["lr_gamma"])

    train = Trainer(model, optimizer, scheduler, train_loader, val_loader, config["train"])
    train.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="input json config")
    args, _ = parser.parse_known_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    main(config)
