import os
import glob
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SHOES_IMAGE_DIR, SHOES_ANNOTATION_DIR

# files in SHOES_ANNOTATION_DIR (from the Shoes download)
TRAIN_SPLIT_FILE = "train_im_names.txt"
EVAL_SPLIT_FILE = "eval_im_names.txt"
ANNOTATION_FILE = "relative_captions_shoes.json"

################################################################################
# *** Helper functions
################################################################################

def read_split(filename):
    with open(os.path.join(SHOES_ANNOTATION_DIR, filename), "r") as f:
        content = f.read().splitlines()
    return content


def read_annot(filename):
    with open(os.path.join(SHOES_ANNOTATION_DIR, filename), "r") as f:
        ann = json.loads(f.read())
    return ann


def write_annot(filename, json_content):
    with open(os.path.join(SHOES_ANNOTATION_DIR, filename), "w") as f:
        json.dump(json_content, f)
    print("Created file:",  os.path.join(SHOES_ANNOTATION_DIR, filename))


################################################################################
# *** Format data into train/validation splits & triplet annotation files, as
# for FashionIQ
################################################################################

def produce_split_files_with_img_full_name():
    """
    Create new versions of "train_im_names.txt" and "eval_im_names.txt" to
    account for the data organization on disk. ie: in these new versions, we use
    the image full name, to avoid having to reorganize the images in the
    directories etc. NOTE: in the produced files, the image paths are given
    relatively to SHOES_IMAGE_DIR
    """

    train_split = read_split(TRAIN_SPLIT_FILE)
    eval_split = read_split(EVAL_SPLIT_FILE)

    train_split_real = []
    eval_split_real = []

    imgfiles_full = [f for f in glob.glob(os.path.join(SHOES_IMAGE_DIR, "womens/*/*/*.jpg"), recursive=True)]
    imgfiles_raw = [os.path.basename(f) for f in imgfiles_full]

    for img in train_split:
        ind = imgfiles_raw.index(img)
        train_split_real.append(os.path.relpath(imgfiles_full[ind], SHOES_IMAGE_DIR))
    print("\nNumber of images in the train split: ", len(train_split_real))
    write_annot("split.train.json", train_split_real)
    
    for img in eval_split:
        ind = imgfiles_raw.index(img)
        eval_split_real.append(os.path.relpath(imgfiles_full[ind], SHOES_IMAGE_DIR))
    print("Number of images in the validation split: ", len(eval_split_real))
    write_annot("split.val.json", eval_split_real)


def produce_annotations_per_split():

    annot = read_annot(ANNOTATION_FILE)

    train_split = read_split(TRAIN_SPLIT_FILE)
    train_split_real = read_annot("split.train.json")
    
    eval_split = read_split(EVAL_SPLIT_FILE)
    eval_split_real = read_annot("split.val.json")

    d_train = {train_split[i]:train_split_real[i] for i in range(len(train_split))}
    d_eval = {eval_split[i]:eval_split_real[i] for i in range(len(eval_split))}

    train_annot = []
    eval_annot = []

    for ann in annot:

        if (ann["ImageName"] in train_split) and (ann["ReferenceImageName"] in train_split):
            a = {"ImageName":d_train[ann["ImageName"]],
                "ReferenceImageName":d_train[ann["ReferenceImageName"]],
                "RelativeCaption":ann["RelativeCaption"]}
            train_annot.append(a)
        
        elif (ann["ImageName"] in eval_split) and (ann["ReferenceImageName"] in eval_split):
            a = {"ImageName":d_eval[ann["ImageName"]],
                "ReferenceImageName":d_eval[ann["ReferenceImageName"]],
                "RelativeCaption":ann["RelativeCaption"]}
            eval_annot.append(a)

    print("\nNumber of triplets in the train split: ", len(train_annot))
    write_annot("triplet.train.json", train_annot)
    print("Number of triplets in the validation split: ", len(eval_annot))
    write_annot("triplet.val.json", eval_annot)


if __name__ == '__main__':

    print("Formating data similarly to FashionIQ.\n")
    print("Expecting Shoes images in:", SHOES_IMAGE_DIR)
    print("Expecting Shoes annotations in:", SHOES_ANNOTATION_DIR)
    produce_split_files_with_img_full_name()
    produce_annotations_per_split()