import os
import warnings
import numpy as np
import torch
from torch.hub import get_dir


MODEL_NAMES_DIC = {
    'vit_b16': 'B_16',
    'vit_b32': 'B_32',
    'vit_l16': 'L_16',
    'vit_l32': 'L_32'
}


def load_state_dict(model_name, model_dir=None, map_location='cpu'):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if model_name in MODEL_NAMES_DIC:
        model_name = MODEL_NAMES_DIC[model_name]
    else:
        raise NotImplementedError

    fp = f'{os.path.join(model_dir, model_name)}.pth'
    if not os.path.exists(fp):
        raise FileNotFoundError

    return torch.load(fp, map_location=map_location)


def load_pretrained_weights(
    model,
    config,
    model_name=None,
    weights_path=None,
    strict=False,
    verbose=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        verbose (bool): Whether to print on completion
    """
    load_class_token = hasattr(model, 'class_token')
    load_patch_embedding = (hasattr(model, 'patch_embedding') and config.num_channels == 3)
    load_pos_embedding = (hasattr(model, 'positional_embedding'))
    load_encoder_norm = (hasattr(model, 'encoder_norm'))

    # Load or download weights
    if weights_path is None:
        state_dict = load_state_dict(model_name)
    else:
        state_dict = torch.load(weights_path)

    has_class_token = True if 'class_token' in state_dict else False

    # Modifications to load partial state dict
    expected_missing_keys = []
    if 'pre_logits' in state_dict.keys():
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    if 'fc' in state_dict.keys():
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if 'class_token' in state_dict.keys() and not load_class_token:
        expected_missing_keys += ['class_token']
    if 'patch_embedding.weight' in state_dict.keys() and not load_patch_embedding:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if 'positional_embedding.pos_embedding' in state_dict.keys() and not load_pos_embedding:
        expected_missing_keys += ['positional_embedding.pos_embedding']
    if 'norm.weight' in state_dict.keys() and not load_encoder_norm:
        expected_missing_keys += ['norm.weight', 'norm.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    modify_dic(state_dict)

    # Change size of positional embeddings
    if load_pos_embedding:
        resize_emb = state_dict['positional_embedding.pos_embedding'].shape != \
            model.state_dict()['positional_embedding.pos_embedding'].shape
        posemb = state_dict['positional_embedding.pos_embedding']
        if resize_emb:
            posemb_new = model.state_dict()['positional_embedding.pos_embedding']
            state_dict['positional_embedding.pos_embedding'] = \
                resize_pos_embedding(posemb=posemb, posemb_new=posemb_new,
                                     has_class_token=has_class_token,
                                     load_class_token=load_class_token)
            maybe_print('Resized positional embeddings from {} to {}'.format(
                        posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        for key in ret.missing_keys:
            assert key in expected_missing_keys, '''
            Missing keys when loading pretrained weights: {}
            Expected missing keys: {}
            '''.format(ret.missing_keys, expected_missing_keys)
        assert not ret.unexpected_keys, \
            '''Unexpected keys when loading pretrained weights: {}
            '''.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('''Missing keys when loading pretrained weights: {}
            Expected missing keys: {}
            '''.format(ret.missing_keys, expected_missing_keys), verbose)
        maybe_print('''Unexpected keys when loading pretrained weights: {}
            '''.format(ret.unexpected_keys), verbose)
        maybe_print('Loaded pretrained weights.', verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def modify_dic(state_dict):
    temp = dict.fromkeys(state_dict.keys(), [])
    for key in temp:
        new_key = None
        if 'norm.' in key:
            new_key = key.replace('norm.', 'encoder_norm.')
        elif 'transformer' in key:
            new_key = key.replace('transformer', 'encoder')

        if new_key:
            state_dict[new_key] = state_dict.pop(key)


def resize_pos_embedding(posemb, posemb_new, has_class_token=True,
                         load_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if load_class_token:
        ntok_new -= 1
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    else:
        posemb_tok, posemb_grid = posemb[:, :], posemb[1:]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    if load_class_token:
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    else:
        posemb = posemb_grid
    return posemb
