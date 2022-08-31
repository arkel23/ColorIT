import os
import re
import argparse
import glob
import imageio


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_progress_video(args):

    # filepaths
    file_path = os.path.join(args.images_path, f'{args.images_names}*.png')
    filenames = glob.glob(file_path)
    filenames.sort(key=natural_keys)

    images = []

    for filename in filenames:
        images.append(imageio.imread(filename))
        if args.print_fns:
            print(filename)

    fp = os.path.join(args.results_dir, f'{args.save_name}.{args.ext}')
    imageio.mimsave(fp, images, fps=args.fps)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='path to folder with images')
    parser.add_argument('--images_names', type=str, default='test')
    parser.add_argument('--results_dir', type=str, default='visualization')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--ext', type=str, default='mp4', choices=('mp4', 'gif'))
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--print_fns', action='store_true')
    args = parser.parse_args()
    args.dataset_root_path = os.path.split(os.path.normpath(args.images_path))[0]

    os.makedirs(args.results_dir, exist_ok=True)
    if not args.save_name:
        os.path.split(os.path.normpath(args.images_path))[0]
        args.save_name = os.path.split(os.path.normpath(args.images_path))[1]

    make_progress_video(args)


main()
