# Experiments on using Global-Local Similarity-Based Anchor Cropping

# Setup

```
pip install -e . 
```

# Usage
```
import torch
from glsim.model_utils import ViT, ViTConfig

model_name = 'vit_b16'
cfg = ViTConfig(model_name, classifier='cls', sim_metric='kld_logo_a', anchor_size=160, 
    aligner='cls', aligner_pos_embedding=True,
    aggregator=True, aggregator_norm=True, aggregator_num_hidden_layers=2)
model = ViT(cfg)

x = torch.rand(2, cfg.num_channels, cfg.image_size, cfg.image_size)
out = model(x)
```


