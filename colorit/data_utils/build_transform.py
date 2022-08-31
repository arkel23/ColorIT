from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from .augmentations import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from .sketch_transforms import desaturate, decontrast, xdog, xdog_serial


MEANS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.485, 0.456, 0.406),
    'tinyin': (0.4802, 0.4481, 0.3975),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5070, 0.4865, 0.4409),
    'svhn': (0.4377, 0.4438, 0.4728),
    'cub': (0.3659524, 0.42010019, 0.41562049)
}

STDS = {
    '05': (0.5, 0.5, 0.5),
    'imagenet': (0.229, 0.224, 0.225),
    'tinyin': (0.2770, 0.2691, 0.2821),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'svhn': (0.1980, 0.2010, 0.1970),
    'cub': (0.07625843, 0.04599726, 0.06182727)
}


class ApplyTransform():
    def __init__(self, is_train, args, transform):
        self.args = args
        self.t = transform
        self.t_tensor_norm = self.get_t_tensor_norm(args)

        self.train_refine_steps = args.train_refine_steps

        self.xdog_sigma_steps = int(self.args.xdog_sigma_percent * self.train_refine_steps)

        self.xdog_phi_steps = int(self.args.xdog_phi_percent * self.train_refine_steps)
        self.xdog_phi_end_step = self.xdog_phi_steps + self.xdog_sigma_steps

        self.decontrast_steps = int(self.args.decontrast_percent * self.train_refine_steps)
        self.decontrast_end_step = self.decontrast_steps + self.xdog_phi_end_step

        self.desaturate_steps = int(self.args.desaturate_percent * self.train_refine_steps)

        self.is_train = is_train

        self.serial = False
        self.sigma = 0.8
        self.sigma_serial = 0.5

        self.k = 1.6
        self.k_serial = 4.5

        self.sigma_low = 0.0
        self.sigma_high = 0.9

        self.phi_low = 40
        self.phi_high = 200

        self.dc_low = 0.4
        self.dc_high = 1.0

        self.ds_low = 0.0
        self.ds_high = 1.0

    def __call__(self, x):
        if self.is_train:
            r = np.random.rand(1)
            if r > 0.5:
                self.serial = True

            # results should come in pairs: b, train_refine_steps, 2, c, h, w or b, train_refine_steps * 2, c, h, w
            transforms = []

            x_0 = self.t(x)
            x_input = self.t_tensor_norm(self.diffusion_transform(x_0, step=0))

            for step in range(self.train_refine_steps - 1):
                x_gt = self.t_tensor_norm(self.diffusion_transform(x_0, step=step+1))
                transforms.append(torch.stack([x_input, x_gt], dim=-1))
                x_input = x_gt

            transforms.append(torch.stack([x_input, self.t_tensor_norm(x_0)], dim=-1))
            return transforms

        else:
            x_gt = self.t(x)
            x_input = self.t_tensor_norm(self.diffusion_transform(x_gt, step=0))
            return torch.stack([x_input, self.t_tensor_norm(x_gt)], dim=-1)

    def diffusion_transform(self, x, step):
        if step < self.xdog_sigma_steps:
            if step == 0:
                w, h = x.size
                x = Image.new('RGB', (w, h), (255, 255, 255))
                return x

            step_variable = self.get_step_variable(
                step, [self.sigma_low, self.sigma_high], self.xdog_sigma_steps, quadratic=True)
            self.sigma = step_variable
            if self.serial:
                x = xdog_serial(x, sigma=self.sigma)
            else:
                x = xdog(x, sigma=self.sigma)

        elif step >= self.xdog_sigma_steps and step < self.xdog_phi_end_step:
            step_variable = self.get_step_variable(
                step - self.xdog_sigma_steps, [self.phi_low, self.phi_high], self.xdog_phi_steps)
            x = xdog(x, sigma=self.sigma, phi=step_variable)

        elif step >= self.xdog_phi_end_step and step < self.decontrast_end_step:
            step_variable = self.get_step_variable(
                step - self.xdog_phi_end_step, [self.dc_low, self.dc_high], self.decontrast_steps)
            x = desaturate(x, 0)
            x = decontrast(x, step_variable)

        elif step >= self.decontrast_end_step:
            step_variable = self.get_step_variable(
                step - self.decontrast_end_step, [self.ds_low, self.ds_high], self.desaturate_steps)
            x = desaturate(x, step_variable)

        return x

    def get_step_variable(self, step, ranges, total_steps, r_mean=0, r_std=0.1, quadratic=False):
        percent = np.linspace(ranges[0], ranges[1], total_steps)[step]
        if quadratic:
            percent = percent ** 2
        r = np.random.normal(r_mean, r_std, 1)[0]
        step_variable = percent + r
        if step_variable >= 0:
            return step_variable
        return percent + abs(r)

    def get_t_tensor_norm(self, args):
        mean = MEANS['imagenet']
        std = STDS['imagenet']
        if args.custom_mean_std:
            mean = MEANS[args.dataset_name] if args.dataset_name in MEANS.keys() else MEANS['05']
            std = STDS[args.dataset_name] if args.dataset_name in STDS.keys() else STDS['05']

        to_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=mean, std=std)

        t = []
        t.append(to_tensor)
        t.append(norm)

        transform = transforms.Compose(t)
        print('Finished transform:', transform)
        return transform


def standard_transform(args, is_train):
    image_size = args.image_size
    resize_size = args.resize_size
    test_resize_size = args.test_resize_size

    if 'cifar' in args.dataset_name and image_size == 32:
        aa = CIFAR10Policy()
    elif args.dataset_name == 'svhn' and image_size == 32:
        aa = SVHNPolicy()
    else:
        aa = ImageNetPolicy()

    t = []

    if is_train:
        if args.random_resized_crop:
            t.append(transforms.RandomResizedCrop(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC))
        elif args.square_resize_random_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomCrop(image_size))
        elif args.short_side_resize_random_crop:
            t.append(transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.RandomCrop((image_size, image_size)))
        elif args.square_center_crop:
            t.append(transforms.Resize(
                (resize_size, resize_size),
                interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(image_size))

        if args.horizontal_flip:
            t.append(transforms.RandomHorizontalFlip())
        if args.vertical_flip:
            t.append(transforms.RandomVerticalFlip())
        if args.jitter_prob > 0:
            t.append(transforms.RandomApply([transforms.ColorJitter(
                brightness=args.jitter_bcs, contrast=args.jitter_bcs,
                saturation=args.jitter_bcs, hue=args.jitter_hue)], p=args.jitter_prob))
        if args.greyscale > 0:
            t.append(transforms.RandomGrayscale(p=args.greyscale))
        if args.blur > 0:
            t.append(transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=args.blur))
        if args.solarize_prob > 0:
            t.append(transforms.RandomApply(
                [transforms.RandomSolarize(args.solarize, p=args.solarize_prob)]))
        if args.aa:
            t.append(aa)
        if args.randaug:
            t.append(transforms.RandAugment())
        if args.trivial_aug:
            t.append(transforms.TrivialAugmentWide())
    else:
        if ((args.dataset_name in ['cifar10', 'cifar100', 'svhn'] and image_size == 32)
           or (args.dataset_name == 'tinyin' and image_size == 64)):
            t.append(transforms.Resize(image_size))
        else:
            t.append(transforms.Resize(
                (test_resize_size, test_resize_size),
                interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(image_size))

    transform = transforms.Compose(t)
    print('Unfinished transform:', transform)
    return transform


def build_transform(args, split):
    is_train = True if split == 'train' else False

    transform = standard_transform(args, is_train)

    transform = ApplyTransform(is_train, args, transform)
    return transform
