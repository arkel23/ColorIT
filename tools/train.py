import os
import time
import random

import wandb
from timm.optim import create_optimizer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from colorit.data_utils.build_dataloaders import build_dataloaders
from colorit.model_utils.build_model import build_model
from colorit.other_utils.build_args import parse_train_args
from colorit.train_utils.misc_utils import summary_stats, stats_test, set_random_seed
from colorit.train_utils.scheduler import build_scheduler
from colorit.train_utils.trainer import Trainer


def adjust_args_general(args):
    if args.attention == 'mixer':
        args.model_name = args.model_name.replace('vit', 'mixer')
    args.run_name = '{}_{}_{}'.format(
        args.dataset_name, args.model_name, args.serial
    )

    args.results_dir = os.path.join(args.results_dir, args.run_name)


def build_environment(args):
    if args.ckpt_path:
        ignore_list = ['ckpt_path', 'transfer_learning', 'test_only', 'batch_size', 'distributed']
        args_temp = vars(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['config'])
        for k, v in args_temp.items():
            if k not in ignore_list:
                setattr(args, k, v)

    if args.serial is None:
        args.serial = random.randint(0, 1000)
    # Set device and random seed
    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model and criterion
    model = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.zero_grad()

    if args.ls:
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()

    # loss and optimizer
    optimizer = create_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, train_loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if not args.ckpt_path:
        adjust_args_general(args)
        os.makedirs(args.results_dir, exist_ok=True)

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader


def main():
    time_start = time.time()

    args = parse_train_args()

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = build_environment(args)

    trainer = Trainer(args, model, criterion, optimizer, lr_scheduler,
                      train_loader, val_loader, test_loader)

    if args.test_only:
        wandb.init(config=args, project=args.project_name)
        wandb.run.name = args.run_name
        time_start = time.time()
        print(args, model.cfg)

        test_loss, max_memory, no_params = trainer.test()

        time_total = time.time() - time_start
        stats_test(test_loss, max_memory, no_params, time_total, args.num_images_test)
        wandb.finish()
    else:
        if args.local_rank == 0:
            if not args.debugging:
                wandb.init(config=args, project=args.project_name)
                wandb.run.name = args.run_name
            if not args.distributed:
                print(model, model.cfg)
            print(args)

        best_loss, best_epoch, max_memory, no_params = trainer.train()

        # summary stats
        if args.local_rank == 0 and not args.debugging:
            time_total = time.time() - time_start
            summary_stats(args.epochs, time_total, best_loss, best_epoch, max_memory, no_params)
            wandb.finish()


if __name__ == '__main__':
    main()
