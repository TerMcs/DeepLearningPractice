import torch
import numpy as np
import random
import hydra

from omegaconf import DictConfig

from utils import get_mnist, train_loop, test_loop
from models.fullyconnected import FullyConnected

@hydra.main(config_path="./config/", config_name="configs.yaml")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)


    model = FullyConnected(input_size=cfg.input_size,
                           hidden_size=cfg.hidden_size,
                           num_classes=cfg.num_classes
                           )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_data, test_data = get_mnist(batch_size=cfg.batch_size)

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_data, model, loss_fn, optimizer)

    #if (epoch + 1) % 20 == 0:
        #curr_lr /= 3
        #update_lr(optimizer, curr_lr)

        test_loop(test_data, model, loss_fn)

if __name__ == "__main__":
    main()
