import argparse
import glob
import os

"""
Check Dataset path with train/val folder
    python checkDataset.py --dataset DATASET_DIR
"""


def main(image_dir, label_dir, dataset_mode):
    if dataset_mode == "train":
        image_mode_dir = image_dir + "/" + dataset_mode
        im_fpath = glob.glob(image_mode_dir + "/*.png")
        label_mode_dir = label_dir + "/" + dataset_mode
        num_train = 0
        for i in im_fpath:
            lb_fn = os.path.splitext(i.split('/')[-1])[0][0:-12] + "_gtFine_color.mat"
            lab_fpath = label_mode_dir + "/" + lb_fn
            if not os.path.exists(lab_fpath):
                print("path not found")
                print(lab_fpath)
                break
            num_train = num_train + 1
        print("All path are found")
        print("training sample : %d" % num_train)
    elif dataset_mode == "val":
        image_mode_dir = image_dir + "/" + dataset_mode
        im_fpath = glob.glob(image_mode_dir + "/*.png")
        label_mode_dir = label_dir + "/" + dataset_mode
        num_val = 0
        for i in im_fpath:
            lb_fn = os.path.splitext(i.split('/')[-1])[0][0:-12] + "_gtFine_color.mat"
            lab_fpath = label_mode_dir + "/" + lb_fn
            if not os.path.exists(lab_fpath):
                print("path not found")
                print(lab_fpath)
                break
            num_val = num_val + 1
        print("All path are found")
        print("validation sample : %d" % num_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Segmentation')
    parser.add_argument('--image', type=str, required=True, help='Specify the directory of image')
    parser.add_argument('--label', type=str, required=True, help='Specify the directory of label')
    parser.add_argument('--mode', type=str, required=True, help='Specify the dataset_mode')
    args = parser.parse_args()
    main(image_dir=args.image, label_dir=args.label, dataset_mode=args.mode)
