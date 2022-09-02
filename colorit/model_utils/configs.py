ATTENTIONS = ('vanilla', 'mixer', 'conv')
DEPATCHIFIERS = (
    'inter_upsample_conv', 'inter_upsample_csse_conv', 'inter_upsample_csse_conv_conv',
    'transconv_single', 'transconv_mult',
    'upsample_conv_single', 'upsample_conv_mult',
    'transconv_ucatconv_single', 'transconv_ucatconv_mult',
    'transconv_ucat_single', 'transconv_ucat_mult',
    'upsample_conv_ucatconv_single', 'upsample_conv_ucatconv_mult',
    'upsample_conv_ucat_single', 'upsample_conv_ucat_mult',
    'transconv_csse_single', 'transconv_csse_mult',
    'transconv_csse_conv_single', 'transconv_csse_conv_mult',
    'upsample_conv_csse_single', 'upsample_conv_csse_mult',
    'upsample_conv_csse_conv_single', 'upsample_conv_csse_conv_mult',
)


class ViTConfig():
    def __init__(self,
                 model_name: str = 'vit_b16',
                 debugging: bool = None,
                 image_size: int = None,
                 patch_size: tuple() = None,
                 slide_step: int = None,
                 num_channels: int = None,

                 depatchifier: str = None,
                 se: str = None,
                 head_use_tanh: bool = None,

                 attention: str = None,
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
                 print_attr: bool = False,
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
        # number of patches
        if self.slide_step:
            self.gh = ((h - self.fh) // self.slide_step + 1)
            self.gw = ((w - self.fw) // self.slide_step + 1)
        else:
            self.gh, self.gw = h // self.fh, w // self.fw
        # flattened sequence length
        self.seq_len = self.gh * self.gw

    def assertions_corrections(self):
        assert self.attention in ATTENTIONS, f'Choose from {ATTENTIONS}'
        assert self.depatchifier in DEPATCHIFIERS, f'Choose from {DEPATCHIFIERS}'

        if self.depatchifier in ('inter_upsample_conv', 'inter_upsample_csse_conv',
                                 'inter_upsample_csse_conv_conv'):
            self.ret_inter = True
            self.encoder_norm = False

    def __repr__(self):
        return str(vars(self))

    def __str__(self):
        return str(vars(self))


def get_base_config():
    """Base ViT config ViT"""
    return dict(
        debugging=False,
        image_size=224,
        patch_size=(16, 16),
        slide_step=None,
        num_channels=3,

        depatchifier='transconv_single',
        se=None,
        head_use_tanh=False,

        attention='vanilla',
        pos_embedding_type='learned',
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        encoder_norm=False,

        representation_size=768,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.1,
        sd=0.0,
        layer_norm_eps=1e-12,
        hidden_act='gelu',
    )


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = get_base_config()
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_b8_config():
    """Returns the ViT-B/8 configuration."""
    config = get_base_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_s16_config():
    """Returns the ViT-S/16 configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=6,
        representation_size=384,
    ))
    return config


def get_s32_config():
    """Returns the ViT-S/32 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_s8_config():
    """Returns the ViT-S/8 configuration."""
    config = get_s16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t16_config():
    """Returns the ViT-T configuration."""
    config = get_base_config()
    config.update(dict(
        hidden_size=192,
        intermediate_size=768,
        num_attention_heads=3,
        representation_size=192,
    ))
    return config


def get_t32_config():
    """Returns the ViT-T/32 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(32, 32)))
    return config


def get_t8_config():
    """Returns the ViT-T/8 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(8, 8)))
    return config


def get_t4_config():
    """Returns the ViT-T/4 configuration."""
    config = get_t16_config()
    config.update(dict(patch_size=(4, 4)))
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
    ))
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.update(dict(patch_size=(32, 32)))
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
    ))
    config.update(dict(patch_size=(14, 14)))
    return config


CONFIGS = {
    'vit_t4': {
        'config': get_t4_config(),
    },
    'vit_t8': {
        'config': get_t8_config(),
    },
    'vit_t16': {
        'config': get_t16_config(),
    },
    'vit_t32': {
        'config': get_t32_config(),
    },
    'vit_s8': {
        'config': get_s8_config(),
    },
    'vit_s16': {
        'config': get_s16_config(),
    },
    'vit_s32': {
        'config': get_s32_config(),
    },
    'vit_b8': {
        'config': get_b8_config(),
    },
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
