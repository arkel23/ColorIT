import os.path as osp
import sys
from contextlib import suppress

import wandb
import numpy as np
from einops import rearrange
import torch
from torchvision.utils import save_image
from timm.models import model_parameters

from .misc_utils import AverageMeter, count_params_single
from .dist_utils import reduce_tensor, distribute_bn
from .mix import cutmix_data, mixup_data
from .scaler import NativeScaler


class Trainer():
    def __init__(self, args, model, criterion, optimizer, lr_scheduler,
                 train_loader, val_loader, test_loader):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.saved = False
        self.curr_iter = 0

        self.amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress
        self.loss_scaler = NativeScaler() if args.fp16 else None

    def train(self):
        self.best_loss = 1e4
        self.best_epoch = 0
        self.max_memory = 0
        self.no_params = 0
        self.lr_scheduler.step(0)

        for epoch in range(self.args.epochs):
            self.epoch = epoch + 1

            if self.args.distributed or self.args.ra > 1:
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch()

            if self.args.local_rank == 0 and \
                    ((self.epoch % self.args.eval_freq == 0) or (self.epoch == self.args.epochs)):
                val_loss = self.validate_epoch(self.val_loader)

                if self.args.debugging:
                    return None, None, None, None

                self.epoch_end_routine(train_loss, val_loss)

        if self.args.local_rank == 0:
            self.train_end_routine(val_loss)

        return self.best_loss, self.best_epoch, self.max_memory, self.no_params

    def prepare_batch(self, batch):
        images = batch
        if type(images) == list:
            images = torch.stack(images, dim=-1)
            images = rearrange(images, 'b c h w n s -> (s b) c h w n')

        if self.args.distributed:
            images = images.cuda(non_blocking=True)
        else:
            images = images.to(self.args.device, non_blocking=True)
        return images

    def predict(self, images, train=True):
        if self.args.cm or self.args.mu:
            r = np.random.rand(1)
            if r < self.args.mix_prob and train:
                images = self.prepare_mix(images)

        images, images_gt = torch.split(images, [1, 1], dim=-1)
        images = images.squeeze(-1)
        images_gt = images_gt.squeeze(-1)

        with self.amp_autocast():
            output = self.model(images, train)

            if train:
                loss = self.criterion(output, images_gt)
            else:
                loss = self.criterion(output[-1], images_gt)

        if (self.args.save_images and train and (self.curr_iter % self.args.save_images == 0)):
            self.save_images(
                images, images_gt, output, osp.join(self.args.results_dir, f'train_{self.curr_iter}.png'))
            self.saved = False
        elif self.args.save_images and not train and not self.saved:
            self.save_images(
                images, images_gt, output, osp.join(self.args.results_dir, f'test_{self.curr_iter}.png'), train=False)
            self.saved = True

        return output, loss

    def prepare_mix(self, images):
        # cutmix and mixup
        if self.args.cm and self.args.mu:
            switching_prob = np.random.rand(1)
            # Cutmix
            if switching_prob < 0.5:
                images = cutmix_data(images, self.args)
            # Mixup
            else:
                images = mixup_data(images, self.args)
        # cutmix only
        elif self.args.cm:
            slicing_idx, sliced = cutmix_data(images, self.args)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
        # mixup only
        elif self.args.mu:
            images = mixup_data(images, self.args)
        return images

    def backward(self, idx, loss):
        # ===================backward=====================
        if self.args.gradient_accumulation_steps > 1:
            with self.amp_autocast():
                loss = loss / self.args.gradient_accumulation_steps
        if self.loss_scaler is not None:
            self.loss_scaler.scale_loss(loss)

        if (idx + 1) % self.args.gradient_accumulation_steps == 0:
            if self.loss_scaler is not None:
                self.loss_scaler(self.optimizer, clip_grad=self.args.clip_grad,
                                 parameters=model_parameters(self.model))
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step_update(num_updates=self.curr_iter)
        return 0

    def update_meters(self, loss, losses, images):
        # ===================meters=====================
        torch.cuda.synchronize()

        if self.args.distributed:
            reduced_loss = reduce_tensor(loss.data, self.args.world_size)
        else:
            reduced_loss = loss.data
        losses.update(reduced_loss.item(), images.size(0))
        return 0

    def print_results(self, idx, losses, loader, train=True):
        if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
            if train:
                lr_curr = self.optimizer.param_groups[0]['lr']
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'LR: {4}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        self.epoch, self.args.epochs, idx, len(loader), lr_curr,
                        loss=losses))
            else:
                print('Val: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        idx, len(loader), loss=losses))
            sys.stdout.flush()
        return 0

    def train_epoch(self):
        """vanilla training"""
        self.model.train()

        losses = AverageMeter()

        for idx, batch in enumerate(self.train_loader):
            images = self.prepare_batch(batch)
            output, loss = self.predict(images, train=True)

            self.backward(idx, loss)

            self.update_meters(loss, losses, images)

            self.print_results(idx, losses, self.train_loader, train=True)

            self.curr_iter += 1

            if self.args.debugging:
                return None, None

        if self.args.local_rank == 0:
            print(' * Loss {losses.avg:.3f}'.format(losses=losses))

        if self.args.distributed:
            distribute_bn(self.model, self.args.world_size, True)

        self.lr_scheduler.step(self.epoch)

        return round(losses.avg, 3)

    def validate_epoch(self, val_loader):
        """validation"""
        # switch to evaluate mode
        self.model.eval()

        losses = AverageMeter()

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                images = self.prepare_batch(batch)
                output, loss = self.predict(images, train=False)

                self.update_meters(loss, losses, images)

                self.print_results(idx, losses, val_loader, train=True)

                if self.args.debugging:
                    return None, None

        if self.args.local_rank == 0:
            print(' * Loss {losses.avg:.3f}'.format(losses=losses))

        return round(losses.avg, 3)

    def epoch_end_routine(self, train_loss, val_loss):
        lr_curr = self.optimizer.param_groups[0]['lr']
        print("Training...Epoch: {} | LR: {}".format(self.epoch, lr_curr))
        log_dic = {'epoch': self.epoch, 'lr': lr_curr,
                   'train_loss': train_loss, 'val_loss': val_loss}
        # if hasattr(self, 'samples'):
        #    log_dic['samples'] = wandb.Image(self.samples)
        wandb.log(log_dic)

        # save the best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.save_model(self.best_epoch, val_loss, mode='best')
        # regular saving
        if self.epoch % self.args.save_freq == 0:
            self.save_model(self.epoch, val_loss, mode='epoch')

    def train_end_routine(self, val_loss):
        # save last
        self.save_model(self.epoch, val_loss, mode='last')
        # VRAM and No. of params
        self.computation_stats()

    def computation_stats(self):
        # VRAM memory consumption
        self.max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        # summary stats
        self.no_params = count_params_single(self.model)

    def save_model(self, epoch, loss, mode):
        state = {
            'config': self.args,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'loss': loss,
            'optimizer': self.optimizer.state_dict(),
        }

        if mode == 'best':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_best.pth')
            print('Saving the best model!')
            torch.save(state, save_file)
        elif mode == 'epoch':
            save_file = osp.join(self.args.results_dir, f'ckpt_epoch_{epoch}.pth')
            print('==> Saving each {} epochs...'.format(self.args.save_freq))
            torch.save(state, save_file)
        elif mode == 'last':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_last.pth')
            print('Saving last epoch')
            torch.save(state, save_file)

    def test(self):
        print(f'Evaluation on test dataloader: ')
        self.epoch = self.args.epochs
        test_loss = self.validate_epoch(self.test_loader)
        self.computation_stats()
        return test_loss, self.max_memory, self.no_params

    def save_images(self, images, images_gt, output, fp, train=True):
        with torch.no_grad():
            if not train:
                samples = torch.stack([images] + [images_gt] + output, dim=-1)
            else:
                samples = torch.stack([images, images_gt, output], dim=-1)
            samples = rearrange(samples, 'b c h w n -> b c h (n w)')
            samples = (samples.data + 1) / 2.0
            save_image(samples, fp, nrow=int(np.sqrt(samples.shape[0])))
        return 0
