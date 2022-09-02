import os
import argparse
import torch

from .yaml_config_hook import yaml_config_hook


VITS = ['vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
        'vit_b8', 'vit_b16', 'vit_bs16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']
MODELS = VITS
ATTENTIONS = ('vanilla', 'mixer', 'conv')
DEPATCHIFIERS = (
    'transconv', 'transconv_ucatconv', 'transconv_se',
    'upsample_conv', 'upsample_conv_ucatconv', 'upsample_conv_se',
    'inter_transconv_conv', 'inter_transconv_se',
    'inter_upsample_conv', 'inter_upsample_conv_conv', 'inter_upsample_conv_se'
)
SES = (None, 'ssce', 'ssce_w', 'csse', 'csse_w', 'cse', 'cse_w')


def add_adjust_common_dependent(args):
    if args.test_only:
        print('When using test only changes batch size to 1 to simulate streaming')
        args.batch_size = 1
        print('When using test only automatically turns off debugging')
        args.debugging = False

    if args.debugging_extreme:
        args.batch_size = 1

    if args.base_lr:
        args.lr = args.base_lr * (args.batch_size / 256)

    args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    # distributed
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(args.device)

        args.effective_batch_size = args.effective_batch_size * args.world_size

        if args.base_lr:
            args.lr = args.base_lr * ((args.world_size * args.batch_size) / 256)

    if not args.resize_size:
        if args.image_size == 32:
            args.resize_size = 36
        elif args.image_size == 64:
            args.resize_size = 72
        else:
            if args.dataset_name == 'ncfm':
                args.resize_size = args.image_size + 32
            elif args.dataset_name in ('aircraft', 'cars', 'dogs', 'pets', 'flowers', 'food'):
                args.resize_size = int(args.image_size / 0.875)
            else:
                args.resize_size = int(args.image_size * 1.34)

    if not args.test_resize_size:
        args.test_resize_size = args.resize_size

    if not (args.random_resized_crop or args.square_resize_random_crop or
            args.short_side_resize_random_crop or args.square_center_crop):
        print('Needs at least one crop, using square_resize_random_crop by default')
        args.square_resize_random_crop = True

    args.train_refine_steps = args.sigma_steps + args.xdog_stg2_steps + args.dc_steps + args.ds_steps

    if not args.infer_refine_steps:
        args.infer_refine_steps = args.train_refine_steps

    return args


def add_common_args():
    parser = argparse.ArgumentParser('Arguments for code: ColorIT')
    # general
    parser.add_argument('--project_name', type=str, default='ColorIT',
                        help='project name for wandb')
    parser.add_argument('--debugging', action='store_true',
                        help='when true disables wandb and exits after a single pass')
    parser.add_argument('--debugging_extreme', action='store_true',
                        help='when true does train but with only a single example')
    parser.add_argument('--serial', default=1, type=int, help='serial number for run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int,
                        default=100, help='print frequency (iters)')
    parser.add_argument('--save_freq', type=int,
                        default=1000, help='save frequency (epochs)')
    parser.add_argument('--results_dir', type=str, default='results_train',
                        help='dir to save models from base_path')
    return parser


def add_data_args(parser):
    parser.add_argument('--image_size', type=int, default=32, help='image_size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--cpu_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--custom_mean_std', action='store_true', help='custom mean/std')
    parser.add_argument('--deit_recipe', action='store_true', help='use deit augs')
    parser.add_argument('--pin_memory', action='store_false', help='pin memory for gpu (def: true)')
    # dataset
    parser.add_argument('--dataset_name', default=None, type=str, help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default=None,
                        help='the root directory for where the data/feature/label files are')
    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')
    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default=None,
                        help='the df csv with classnames and class ids, root/classid_classname.csv')
    return parser


def add_optim_scheduler_args(parser):
    # optimizer
    parser.add_argument('--opt', default='adamw', type=str,
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--base_lr', type=float, default=None,
                        help='base_lr if using scaling lr (lr = base_lr * bs/256')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # fp 16 stability and gradient accumulation
    parser.add_argument('--fp16', action='store_false', help='use fp16 (on by default)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate before backward/update pass.')
    # lr scheduler
    parser.add_argument('--sched', default='cosine', type=str,
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--t_in_epochs', action='store_true',
                        help='update per epoch (instead of per iter)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None,
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67,
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0,
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay_epochs', type=float, default=30,
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='steps to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=5,
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10,
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1,
                        help='LR decay rate (default: 0.1)')
    return parser


def add_vit_args(parser):
    # default config is based on vit_b16 but default model is vit_t4
    parser.add_argument('--slide_step', type=int, default=None, help='use sliding patchifier')
    # encoder related
    parser.add_argument('--attention', type=str, choices=ATTENTIONS,
                        help='use global attention or a mix of locan and global')
    parser.add_argument('--pos_embedding_type', type=str, default=None,
                        help='positional embedding for encoder, def: learned')
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--intermediate_size', type=int, default=None)
    parser.add_argument('--num_attention_heads', type=int, default=None)
    parser.add_argument('--num_hidden_layers', type=int, default=None)
    parser.add_argument('--encoder_norm', action='store_true',
                        help='norm after encoder')
    # transformers in general
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=None)
    parser.add_argument('--hidden_dropout_prob', type=float, default=None)
    parser.add_argument('--layer_norm_eps', type=float, default=None)
    parser.add_argument('--hidden_act', type=str, default=None)
    return parser


def add_augmentation_args(parser):
    # cropping
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size before cropping')
    parser.add_argument('--test_resize_size', type=int, default=None, help='test resize_size before cropping')
    parser.add_argument('--random_resized_crop', action='store_true',
                        help='crop random aspect ratio then resize to square')
    parser.add_argument('--square_resize_random_crop', action='store_true',
                        help='resize first to square then random crop')
    parser.add_argument('--short_side_resize_random_crop', action='store_true',
                        help='resize first so short side is resize_size then random crop a square')
    parser.add_argument('--square_center_crop', action='store_true',
                        help='resize first to square then center crop when training')
    # flips
    parser.add_argument('--horizontal_flip', action='store_true',
                        help='use horizontal flip when training (on by default)')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='use vertical flip (off by default)')
    # augmentation policies
    parser.add_argument('--aa', action='store_true', help='Auto augmentation used')
    parser.add_argument('--randaug', action='store_true', help='RandAugment augmentation used')
    parser.add_argument('--trivial_aug', action='store_true', help='use trivialaugmentwide')
    # color and distortion
    parser.add_argument('--jitter_prob', type=float, default=0.0,
                        help='color jitter probability of applying (0.8 for simclr)')
    parser.add_argument('--jitter_bcs', type=float, default=0.4,
                        help='color jitter brightness contrast saturation (0.4 for simclr)')
    parser.add_argument('--jitter_hue', type=float, default=0.1,
                        help='color jitter hue value (0.1 for simclr)')
    parser.add_argument('--blur', type=float, default=0.0,
                        help='gaussian blur probability (0.5 for simclr)')
    parser.add_argument('--greyscale', type=float, default=0.0,
                        help='gaussian blur probability (0.2 for simclr)')
    parser.add_argument('--solarize_prob', type=float, default=0.0,
                        help='solarize transform probability (0.2 for byol if image_size>32)')
    parser.add_argument('--solarize', type=int, default=128,
                        help='solarize pixels with higher value than (def: 128)')
    # cutmix, mixup, random erasing
    parser.add_argument('--cm', action='store_true', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_true', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    parser.add_argument('--re', default=0.0, type=float,
                        help='Random Erasing probability (def: 0.25)')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    # regularization
    parser.add_argument('--ra', type=int, default=0, help='repeated augmentation (def: 3)')
    parser.add_argument('--sd', default=0.0, type=float,
                        help='rate of stochastic depth (def: 0.1)')
    parser.add_argument('--ls', action='store_true', help='label smoothing')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    return parser


def add_diffusion_args(parser):
    parser.add_argument('--head_use_tanh', action='store_true',
                        help='if use then adds tanh after depatchifier')
    parser.add_argument('--depatchifier', type=str, default='transconv',
                        choices=DEPATCHIFIERS, help='method for depatchifying')
    parser.add_argument('--se', type=str, default=None,
                        choices=SES, help='se for depatchifying')
    parser.add_argument('--se_ratio', type=float, default=4,
                        help='compression/expansion ratio for se block')
    parser.add_argument('--se_residual', action='store_true',
                        help='if true then result of se is added as a residual')
    parser.add_argument('--se_reweight_target', action='store_true',
                        help='if true then reweights target based on context (original image)')
    # sketching / drawing / colorization augs
    parser.add_argument('--infer_refine_steps', type=int, default=None,
                        help='number of steps for refinement during inference (like diffusion)')
    parser.add_argument('--sigma_steps', type=int, default=0)
    parser.add_argument('--xdog_stg2_steps', type=int, default=0)
    parser.add_argument('--dc_steps', type=int, default=0)
    parser.add_argument('--ds_steps', type=int, default=0)
    parser.add_argument('--xdog_serial', action='store_true',
                        help='if true then uses xdog serial instead of xdog')
    parser.add_argument('--xdog_random', action='store_true',
                        help='if true then alternates between xdog and xdog_serial')
    return parser


def parse_train_args(ret_parser=False):
    parser = add_common_args()
    parser.add_argument('--save_images', type=int, default=100,
                        help='save images every x iterations')
    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')
    parser.add_argument('--test_only', action='store_true',
                        help='when true skips training and evaluates model on test dataloader')
    parser.add_argument('--eval_freq', type=int, default=10, help='eval every x epochs')
    # models in general
    parser.add_argument('--model_name', type=str, default='vit_t4', choices=MODELS)
    parser.add_argument('--pretrained', action='store_true', help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to custom pretrained ckpt')
    parser.add_argument('--transfer_learning', action='store_true',
                        help='not load fc layer when using custom ckpt')
    # distributed
    parser.add_argument('--dist_eval', action='store_true',
                        help='validate using dist sampler (else do it on one gpu)')
    parser = add_vit_args(parser)
    parser = add_diffusion_args(parser)
    parser = add_data_args(parser)
    parser = add_optim_scheduler_args(parser)
    parser = add_augmentation_args(parser)
    parser.add_argument("--cfg", type=str,
                        help="If using it overwrites args and reads yaml file in given path")
    parser.add_argument("--cfg_is_bs", type=str,
                        help="If using it overwrites args and reads yaml file in given path")
    parser.add_argument("--cfg_method", type=str,
                        help="If using it overwrites args and reads yaml file in given path")

    if ret_parser:
        return parser

    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)

    return args


def parse_inference_args():
    parser = parse_train_args(ret_parser=True)
    parser.add_argument('--images_path', type=str, default='samples',
                        help='path to folder (with images) or image')
    parser.add_argument('--results_inference', type=str, default='results_inference',
                        help='path to folder to save result crops')
    parser.add_argument('--save_crops_only', action='store_true',
                        help='save only crop')
    args = parser.parse_args()
    adjust_config(args)
    args = add_adjust_common_dependent(args)
    return args


def adjust_config(args):
    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.cfg_is_bs:
        config = yaml_config_hook(os.path.abspath(args.cfg_is_bs))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    if args.cfg_method:
        config = yaml_config_hook(os.path.abspath(args.cfg_method))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)


if __name__ == '__main__':
    args = parse_train_args()
    print(args)
