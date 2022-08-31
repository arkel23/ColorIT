from colorit import ViTVanilla, ViTVanillaConfig

# in21k pretrained
models_list = ['vit_b16', 'vit_b32', 'vit_l16', 'vit_l32']
for model_name in models_list:
    cfg = ViTVanillaConfig(model_name=model_name, load_repr_layer=True)
    model = ViTVanilla(cfg, pretrained=True)
    print(cfg)
    # print(model)
