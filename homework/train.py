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
    # if args.continue_training:
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.L1Loss().to(device)

    import inspect
    # transform = args.transform
    transform = dense_transforms.Compose([dense_transforms.ColorJitter[.9, .9, .9, .1],
        dense_transforms.RandomHorizontalFlip(0), dense_transforms.ToTensor()])
    # transform = eval(args.transform,
    #                  {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data',transform=transform, num_workers=4)
    # valid_data = load_data('data/valid', num_workers=4)

    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch: ", epoch)
        model.train()
        # confusion = ConfusionMatrix(len(LABEL_NAMES))
        for img, label in train_data:
            # if train_logger is not None:
            #     train_logger.add_images('augmented_image', img[:4])
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            # confusion.add(logit.argmax(1), label)

            # if train_logger is not None:
            #     train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            # global_step += 1

        # if train_logger:
        #     train_logger.add_scalar('accuracy', confusion.global_accuracy, global_step)
        #     import matplotlib.pyplot as plt
        #     f, ax = plt.subplots()
        #     ax.imshow(confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
        #     for i in range(confusion.per_class.size(0)):
        #         for j in range(confusion.per_class.size(1)):
        #             ax.text(j, i, format(confusion.per_class[i, j], '.2f'),
        #                     ha="center", va="center", color="black")
            # train_logger.add_figure('confusion', f, global_step)

        # model.eval()
        # val_confusion = ConfusionMatrix(len(LABEL_NAMES))
        # for img, label in valid_data:
        #     img, label = img.to(device), label.to(device)
        #     val_confusion.add(model(img).argmax(1), label)

        # if valid_logger:
        #     valid_logger.add_scalar('accuracy', val_confusion.global_accuracy, global_step)
        #     import matplotlib.pyplot as plt
        #     f, ax = plt.subplots()
        #     ax.imshow(val_confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
        #     for i in range(val_confusion.per_class.size(0)):
        #         for j in range(val_confusion.per_class.size(1)):
        #             ax.text(j, i, format(val_confusion.per_class[i, j], '.2f'),
        #                     ha="center", va="center", color="black")
        #     valid_logger.add_figure('confusion', f, global_step)

        # if valid_logger is None or train_logger is None:
        #     print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, confusion.global_accuracy,
        #                                                             val_confusion.global_accuracy))
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














































# def train(args):

#     from os import path
#     model = Planner()
#     # train_logger, valid_logger = None, None
#     # if args.log_dir is not None:
#     #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
#     #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

#     """
#     Your code here, modify your HW3 code
#     """

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     model = Planner().to(device)
#     if args.continue_training:
#         model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

#     # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

#     import inspect
#     transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
#     train_data = load_data('drive_data', num_workers=4, transform=transform)
#     # valid_data = load_detection_data('drive_data/valid', num_workers=4)

#     det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
#     size_loss = torch.nn.MSELoss(reduction='none')

#     global_step = 0
#     for epoch in range(args.num_epoch):
#         print("epoch: ", epoch)
#         model.train()

#         for img, gt_det, gt_size in train_data:
#             img, gt_det, gt_size = img.to(device), gt_det.to(device), gt_size.to(device)
#             size_w, _ = gt_det.max(dim=1, keepdim=True)
#             cls_w = gt_det * 1 + (1-gt_det) * gt_det.mean() ** 0.5

#             det, size = model(img)
#             # Continuous version of focal loss
#             p_det = torch.sigmoid(det * (1-2*gt_det))
#             det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
#             size_loss_val = (size_w * size_loss(size, gt_size)).mean() / size_w.mean()
#             loss_val = det_loss_val + size_loss_val * args.size_weight

#             # if train_logger is not None and global_step % 100 == 0:
#             #     train_logger.add_image('image', img[0], global_step)
#             #     train_logger.add_image('label', gt_det[0], global_step)
#             #     train_logger.add_image('pred', torch.sigmoid(det[0]), global_step)

#             # if train_logger is not None:
#             #     train_logger.add_scalar('det_loss', det_loss_val, global_step)
#             #     train_logger.add_scalar('size_loss', size_loss_val, global_step)
#             #     train_logger.add_scalar('loss', loss_val, global_step)
#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()
#             global_step += 1

#         # if valid_logger is None or train_logger is None:
#         #     print('epoch %-3d' %
#         #           (epoch))

#     save_model(model)


# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--log_dir')
#     # Put custom arguments here
#     parser.add_argument('-n', '--num_epoch', type=int, default=150)
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
#     parser.add_argument('-c', '--continue_training', action='store_true')
#     parser.add_argument('-t', '--transform',
#                         default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
#     parser.add_argument('-w', '--size-weight', type=float, default=0.01)

#     args = parser.parse_args()
#     train(args)
