import random

import numpy as np
import torch
import wandb


def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])


def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


def set_random_seed(seed=0, numpy=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if numpy:
        np.random.seed(seed)
    return 0


def summary_stats(epochs, time_total, best_loss,
                  best_epoch, max_memory, no_params):
    time_avg = round((time_total / epochs) / 60, 2)
    best_time = round((time_avg * best_epoch) / 60, 2)
    time_total = round(time_total / 60, 2)  # mins
    no_params = round(no_params / (1e6), 2)  # millions of parameters
    max_memory = round(max_memory, 2)

    print('''Total run time (minutes): {}
          Average time per epoch (minutes): {}
          Best loss {} at epoch {}. Time to reach this (minutes): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_loss, best_epoch, best_time,
                     max_memory, no_params))

    wandb.run.summary['time_total'] = time_total
    wandb.run.summary['time_avg'] = time_avg
    wandb.run.summary['best_loss'] = best_loss
    wandb.run.summary['best_epoch'] = best_epoch
    wandb.run.summary['best_time'] = best_time
    wandb.run.summary['max_memory'] = max_memory
    wandb.run.summary['no_params'] = no_params
    return 0


def stats_test(test_loss, max_memory, no_params, time_total, num_images):
    throughput = round(num_images / time_total, 2)
    no_params = round(no_params / (1e6), 2)  # millions of parameters
    max_memory = round(max_memory, 2)

    print('''Throughput (images / s): {}
          Test loss (%): {}
          Class deviation (%): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          '''.format(throughput, test_loss, max_memory, no_params))

    wandb.run.summary['test_loss'] = test_loss
    wandb.run.summary['throughput'] = throughput
    wandb.run.summary['max_memory'] = max_memory
    wandb.run.summary['no_params'] = no_params
    return 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
