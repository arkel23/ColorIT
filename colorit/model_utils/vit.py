import torch
from torch import nn
from einops.layers.torch import Rearrange

from .transformer import Transformer
from .depatchifier import Depatchifier
from .load_pretrained_weights import load_pretrained_weights


class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding


class ViT(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.debugging = config.debugging

        # Patch embedding
        stride = (config.slide_step, config.slide_step) if config.slide_step else (config.fh, config.fw)
        self.patchify = nn.Sequential(
            nn.Conv2d(
                in_channels=config.num_channels, out_channels=config.hidden_size,
                kernel_size=(config.fh, config.fw), stride=stride
            ),
            Rearrange('b d gh gw -> b (gh gw) d')
        )

        # Positional embedding
        if config.pos_embedding_type == 'learned':
            self.positional_embedding = LearnedPositionalEmbedding1D(
                config.seq_len, config.hidden_size)

        # Transformer encoder
        self.encoder = Transformer(
            num_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            ff_dim=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            layer_norm_eps=config.layer_norm_eps,
            sd=config.sd,
            attn=config.attention,
            seq_len=config.seq_len,
            ret_inter=hasattr(config, 'ret_inter'))

        if config.encoder_norm:
            self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.depatchifier = Depatchifier(config)

        if config.head_use_tanh:
            self.tanh = nn.Tanh()

        # Initialize weights
        self.init_weights()

        if pretrained:
            load_pretrained_weights(self, config, config.model_name)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        if hasattr(self, 'positional_embedding'):
            nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)

    def forward(self, x):
        """
        x (tensor): b k c fh fw -> b s d
        """
        self.maybe_print('Before tokenizing: ', x.shape)
        x = self.patchify(x)
        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)
        self.maybe_print('After tokenizing: ', x.shape)

        x = self.encoder(x)
        if hasattr(self, 'encoder_norm'):
            x = self.encoder_norm(x)
        # self.maybe_print('After encoder: ', x.shape)

        x = self.depatchifier(x)
        self.maybe_print('After depatchifier: ', x.shape)

        if hasattr(self, 'tanh'):
            x = self.tanh(x)
        return x

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)
