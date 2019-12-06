from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
import torchvision
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Planner().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.L1Loss().to(device)

    import inspect

    transform = dense_transforms.Compose([dense_transforms.ColorJitter(.9, .9, .9, .1),
        dense_transforms.RandomHorizontalFlip(0), dense_transforms.ToTensor()])
   
    train_data = load_data('drive_data',transform=transform, num_workers=4)
   

    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch: ", epoch)
        model.train()

        for img, label in train_data:

            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step() 


        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=75)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ToTensor()])')

    args = parser.parse_args()
    train(args)
