import os
import argparse
import glob
import pandas as pd
import numpy as np
from PIL import Image


def search_images(args):
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


def add_image_to_dics(fp, file_paths):
    # verify the image is RGB
    img = Image.open(fp)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.mode == 'RGB':
        abs_path, filename = os.path.split(fp)
        _, folder_name = os.path.split(abs_path)
        rel_path = os.path.join(folder_name, filename)
        file_paths.append(rel_path)

    return 0


def save_filename_classid(args, file_paths):
    # save dataframe to hold the class IDs and the relative paths of the files
    df = pd.DataFrame(file_paths, columns=['class_id'])
    print(df.head())

    dataset_name = os.path.basename(os.path.normpath(args.images_path))
    df_name = args.save_name if args.save_name else f'{dataset_name}.csv'
    save_path = os.path.join(args.dataset_root_path, df_name)
    df.to_csv(save_path, sep=',', header=True, index=False)


def make_data_dic(args):
    '''
    makes an imagefolder (imagenet style) with images of class in a certain
    folder into a txt dictionary with the first column being the
    file dir (relative) and the second into the class
    '''
    files_all = search_images(args)

    # filename and classid pairs
    file_paths = []

    for fp in files_all:
        add_image_to_dics(fp, file_paths)

    print('Total images files post-filtering (RGB only): ', len(file_paths))

    # save filename_classid df
    save_filename_classid(args, file_paths)


def main():
    '''
    input is the path to the folder with imagenet-like structure
    imagenet/
    imagenet/class1/
    imagenet/class2/
    ...
    imagenet/classN/
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='path to folder like IN')
    parser.add_argument('--save_name', type=str, default=None)
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.images_path))[0]

    make_data_dic(args)


main()
