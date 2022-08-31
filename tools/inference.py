import os
import glob

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
from einops import rearrange

from colorit.other_utils.build_args import parse_inference_args
from colorit.data_utils.build_transform import build_transform

from train import build_environment


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def search_images(args):
    # if path is a file
    if os.path.isfile(args.images_path):
        return [args.images_path]
    # else if directory
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(args.images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(files_all))
    return files_all


def save_images(images_og, images_crops, fp, image_size, resize_og=False, save_crops_only=False):
    with torch.no_grad():
        images_crops = images_crops.reshape(3, image_size, -1)
        if save_crops_only:
            samples = (images_crops.data + 1) / 2.0
        else:
            if resize_og:
                images_og = F.interpolate(images_og, size=(image_size, image_size), mode='bicubic')
            images_og = images_og.reshape(3, image_size, image_size)
            images = torch.cat((images_og, images_crops), dim=2)
            samples = (images.data + 1) / 2.0
        save_image(samples, fp, nrow=1)
        print(f'Saved file to : {fp}')
    return 0


def inference(args):
    files_all = search_images(args)

    model, _, _, _, _, _, _ = build_environment(args)
    model.eval()

    if args.results_inference:
        args.results_dir = os.path.join(args.results_inference, args.model_name)
        os.makedirs(args.results_dir, exist_ok=True)

    transform = build_transform(args=args, split='test')

    # Load class names
    if args.df_classid_classname:
        fp = os.path.join(args.dataset_root_path, args.df_classid_classname)
        dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()

    images_crops = None

    for file in files_all:
        print(file)
        img = prepare_img(file, args, transform)

        # Classify
        with torch.no_grad():
            outputs = model(img)

        if args.classifier_aux and args.anchor_size:
            outputs, output_aux, images_crops = outputs
        elif args.classifier_aux:
            outputs, output_aux = outputs
        elif args.anchor_size or args.model_name == 'cal':
            outputs, images_crops = outputs

        outputs = outputs.squeeze(0)
        for idx in torch.topk(outputs, k=3).indices.tolist():
            prob = torch.softmax(outputs, -1)[idx].item()
            if args.df_classid_classname:
                classname = dic_classid_classname[idx]
                print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100))
            else:
                print('[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100))

        if images_crops is not None:
            if args.model_name == 'cal':
                if args.cal_save_all:
                    images_crops = rearrange(images_crops, 'b c h w k -> b c h (k w)')
                else:
                    images_crops = images_crops[:, :, :, :, 0]

            fn = '{}_crops.png'.format(os.path.splitext(os.path.split(file)[1])[0])
            fp = os.path.join(args.results_dir, fn)
            save_images(img, images_crops, fp, args.image_size,
                        args.pre_resize_factor, args.save_crops_only)

        if args.debugging:
            print('Finished.')
            return 0

    return 0


def main():
    args = parse_inference_args()
    inference(args)
    return 0


if __name__ == '__main__':
    main()
