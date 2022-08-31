import os
import argparse

import pandas as pd


DATASETS = ('cub', 'nabirds', 'aircraft', 'cars', 'dogs', 'pets', 'flowers',
            'food', 'inat2017')


def make_classid_classname(args):
    if args.dataset_name == 'cub':
        # default_name: classes.txt
        df = pd.read_csv(args.classes_path, delimiter=r'\s+', names=['class_id', 'class_name'])
        df['class_id'] = df['class_id'] - 1
    elif args.dataset_name == 'nabirds':
        # default_name: classes.txt
        raise NotImplementedError
        # class_names may have spaces
        df = pd.read_csv(args.classes_path, delimiter=r'\s+', names=['class_id', 'class_name'])
    else:
        raise NotImplementedError
    df.to_csv(os.path.join(args.dataset_root_path, args.save_name), header=True, index=False)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=DATASETS, help='dataset name')
    parser.add_argument('--classes_path', type=str, help='path to filename with classes')
    parser.add_argument('--save_name', type=str, default='classid_classname.csv')
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.classes_path))[0]

    make_classid_classname(args)


main()
