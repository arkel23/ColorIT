import torch
import torch.nn as nn

from .vit import ViT
from .configs import ViTConfig


VITS = (
    'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32', 'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_bs16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14')


def build_model(args):
    # initiates model and loss
    if args.model_name in VITS:
        model = VisionTransformer(args)
    else:
        raise NotImplementedError

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'))['model']
        expected_missing_keys = []
        if args.transfer_learning:
            # modifications to load partial state dict
            if ('model.head.weight' in state_dict):
                expected_missing_keys += ['model.head.weight', 'model.head.bias']
            for key in expected_missing_keys:
                state_dict.pop(key)
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
              Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
        print('Loaded from custom checkpoint.')

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized model: {args.model_name}')
    return model


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        # init default config
        cfg = ViTConfig(model_name=args.model_name)
        # modify config if given an arg otherwise keep defaults
        args_temp = vars(args)
        for k, v in args_temp.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v if v is not None else getattr(cfg, k))
        cfg.assertions_corrections()
        cfg.calc_dims()
        # update the args with the final model config
        for attribute in vars(cfg):
            if hasattr(args, attribute):
                setattr(args, attribute, getattr(cfg, attribute))
        # init model
        self.model = ViT(cfg, pretrained=args.pretrained)
        self.cfg = cfg
        self.infer_refine_steps = args.infer_refine_steps

    def forward(self, images, train=True):
        out = self.model(images)
        if not train:
            outs = []
            for _ in range(self.infer_refine_steps - 1):
                out = self.model(out)
                outs.append(out)
            out = outs
        return out
