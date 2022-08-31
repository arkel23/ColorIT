import os
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_set(args, split, transform=None):
    ds = DatasetImgTarget(args, split=split, transform=transform)

    setattr(args, f'num_images_{split}', ds.__len__())
    print(f"{args.dataset_name} {split} split. N={ds.__len__()}.")
    return ds


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform

        if args.debugging_extreme:
            split = 'train'

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'
        assert os.path.isdir(os.path.join(self.root, self.images_folder)), \
            f'{os.path.join(self.root, self.images_folder)} is not a directory.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.data = self.df['dir'].to_numpy()

        if args.debugging_extreme:
            self.data = self.data[:1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.data[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)
