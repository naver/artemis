#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import re

MAIN_DIR = os.path.dirname(os.path.realpath(__file__)) # default root for vocabulary files, model checkpoints, ranking files, heatmaps

DATA_DIR = f'{MAIN_DIR}/data'
VOCAB_DIR = f'{MAIN_DIR}/vocab'
CKPT_DIR = f'{MAIN_DIR}/ckpt'
RANKING_DIR = f'{MAIN_DIR}/rankings'
HEATMAP_DIR = f'{MAIN_DIR}/heatmaps'

################################################################################
# *** Environment-related configuration
################################################################################

TORCH_HOME = None # where ImageNet's pretrained models (resnet50/resnet18) weights are stored, locally on your machine
GLOVE_DIR = None # where GloVe vectors (`glove.840B.300d.txt.pt`) are stored, locally on your machine

################################################################################
# *** Data paths
################################################################################

# FashionIQ
FASHIONIQ_IMAGE_DIR = f'{DATA_DIR}/fashionIQ/img'
FASHIONIQ_ANNOTATION_DIR = f'{DATA_DIR}/fashionIQ/annotations'

# Shoes
SHOES_IMAGE_DIR = f'{DATA_DIR}/shoes/images'
SHOES_ANNOTATION_DIR = f'{DATA_DIR}/shoes/annotations'

# CIRR
CIRR_IMAGE_DIR = f'{DATA_DIR}/cirr/img_feat_res152'
CIRR_ANNOTATION_DIR = f'{DATA_DIR}/cirr'

# Fashion200k
FASHION200K_IMAGE_DIR = f'{DATA_DIR}/fashion200K'
FASHION200K_ANNOTATION_DIR = f'{FASHION200K_IMAGE_DIR}/labels'

################################################################################
# *** OTHER
################################################################################

# Function to replace "/", "-" and "\" by a space and to remove all other caracters than letters or spaces (+ remove duplicate spaces)
cleanCaption = lambda cap : " ".join(re.sub('[^(a-zA-Z)\ ]', '', re.sub('[/\-\\\\]', ' ', cap)).split(" "))
