import torch
import hydra

from utils import get_cifar10, train_loop, test_loop
from models.resnet import ResNet

@hydra.main(config_path="config/", config_name="configs.yaml")
def main(cfg):

if __name__ == "__main__":
    main()
