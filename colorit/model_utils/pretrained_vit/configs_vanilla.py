HEADS = (None, 'cls', 'pool')


class ViTVanillaConfig():
    def __init__(self,
                 model_name: str = 'vit_b16',
                 image_size: int = None,
                 patch_size: tuple() = None,
                 load_fc_layer: bool = None,
                 load_repr_layer: bool = None,
                 classifier: str = None,
                 num_classes: int = None,
                 num_channels: int = None,

                 pos_embedding_type: str = None,
                 hidden_size: int = None,
                 intermediate_size: int = None,
                 num_attention_heads: int = None,
                 num_hidden_layers: int = None,
                 encoder_norm: bool = None,

                 representation_size: int = None,
                 attention_probs_dropout_prob: float = None,
                 hidden_dropout_prob: float = None,
                 sd: float = None,
                 layer_norm_eps: float = None,
                 hidden_act: str = None,
                 url: str = None,
                 print_attr: bool = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        default = CONFIGS[model_name]

        input_args = locals()
        for k, v in input_args.items():
            if k in default['config'].keys():
                setattr(self, k, v if v is not None else default['config'][k])

        self.assertions_corrections()
        self.calc_dims()
        if print_attr:
            print(vars(self))

    def as_tuple(self, x):
        return x if isinstance(x, tuple) else (x, x)

    def calc_dims(self):
        h, w = self.as_tuple(self.image_size)  # image sizes
        self.fh, self.fw = self.as_tuple(self.patch_size)  # patch sizes
        self.gh, self.gw = h // self.fh, w // self.fw  # number of patches
        self.seq_len = self.gh * self.gw  # sequence length individual image

        if self.classifier == 'cls':
            self.seq_len += 1

    def assertions_corrections(self):
        assert self.classifier in HEADS, f'Choose from {HEADS}'

        if self.image_size == 384 and self.num_classes == 1000:
            self.url = self.url.replace('21k', '21k%2Bimagenet2012')

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        image_size=224,
        patch_size=(16, 16),
        load_fc_layer=True,
        load_repr_layer=False,
        classifier='cls',
        classifier_aux=None,
        num_classes=21843,
        num_channels=3,

        attention='vanilla',
        pos_embedding_type='learned',
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        encoder_norm=True,

        representation_size=768,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        sd=0.0,
        layer_norm_eps=1e-12,
        hidden_act='gelu',
        url=None
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    config.update(dict(
        url='"https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'))
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz'))
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        num_hidden_layers=24,
        representation_size=1024,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz'
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(
        patch_size=(32, 32),
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz'))
    return config


def get_h14_config():
    """Returns the ViT-H/14 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=1280,
        intermediate_size=5120,
        num_attention_heads=16,
        num_hidden_layers=32,
        representation_size=1280,
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz'
    ))
    config.update(dict(patch_size=(14, 14)))
    return config


CONFIGS = {
    'vit_b16': {
        'config': get_b16_config(),
    },
    'vit_b32': {
        'config': get_b32_config(),
    },
    'vit_l16': {
        'config': get_l16_config(),
    },
    'vit_l32': {
        'config': get_l32_config(),
    },
    'vit_h14': {
        'config': get_h14_config(),
    }
}
