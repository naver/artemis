#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import argparse
from config import MAIN_DIR, VOCAB_DIR, CKPT_DIR, RANKING_DIR, HEATMAP_DIR

print("Main directory (default root for vocabulary files, model checkpoints, ranking files, heatmaps...): "+MAIN_DIR)

parser = argparse.ArgumentParser()

# Data paths & directories
parser.add_argument('--data_name', default='fashionIQ', choices=('fashionIQ', 'fashion200K', 'shoes', 'cirr'), help='Dataset name (fashionIQ|fashion200K|shoes|cirr).')
parser.add_argument('--vocab_dir', default=VOCAB_DIR, help='Path to saved vocabulary pickle files')

# Directories for experiment outputs (logging, ckpt, rankings/predictions)
parser.add_argument('--exp_name', default='X', help='Experiment name, used as sub-directory to save experiment-related files (model, ranking files, heatmaps...).')
parser.add_argument('--ckpt_dir', default=CKPT_DIR, help='Directory in which to save the models from the different experiments.')
parser.add_argument('--ranking_dir', default=RANKING_DIR, type=str, help='Directory in which to save the ranking/prediction files, if any to save.')
parser.add_argument('--heatmap_dir', default=HEATMAP_DIR, type=str, help='Directory in which to save the heatmaps.')

# Data/Dataloaders parameters
parser.add_argument('--batch_size', default=32, type=int, help='Size of a mini-batch.')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
parser.add_argument('--categories', default='all', type=str, help='Names of the data categories to consider for a given dataset. Category names must be separated with a space. Specify "all" to consider them all (the interpretation of "all" depends on the dataset).')

# Model settings
parser.add_argument('--model_version', default='ARTEMIS', choices=('cross-modal', 'visual-search', 'late-fusion', 'EM-only', 'IS-only', 'ARTEMIS', "TIRG"), help='Model version (ARTEMIS, one of ARTEMIS ablatives, or TIRG)')
parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='Path of the ckpt to resume from (default: none).')
parser.add_argument('--embed_dim', default=512, type=int, help='Dimensionality of the final text & image embeddings.')
parser.add_argument('--cnn_type', default='resnet50', help='The CNN used as image encoder.')
parser.add_argument('--load_image_feature', default=0, type=int, help="Whether (if int > 0) to load pretrained image features instead of loading raw images and using a cnn backbone. Indicate the size of the feature (int).")
parser.add_argument('--txt_enc_type', default='bigru', choices=('bigru', 'lstm'), help="The text encoder (bigru|lstm).")
parser.add_argument('--lstm_hidden_dim', default=1024, type=int, help='Number of hidden units in the LSTM.')
parser.add_argument('--wemb_type', default='glove', choices=('glove', 'None'), type=str, help='Word embedding (glove|None).')
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.')

# Training / optimizer settings
parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
parser.add_argument('--lr', default=.0005, type=float, help='Initial learning rate.')
parser.add_argument('--step_lr', default=10, type=int, help="Step size, number of epochs after which to apply a learning rate decay.")
parser.add_argument('--gamma_lr', default=0.5, type=float, help="Learning rate decay.")
parser.add_argument('--learn_temperature', action='store_true', help='Whether to use and learn the temperature parameter (the scores given to the loss criterion are multiplied by a trainable version of --temperature).')
parser.add_argument('--temperature', default=2.65926, type=float, help='Temperature parameter.')
parser.add_argument('--validate', default='val', choices=('val', 'test', 'test-val'), help='Split(s) on which the model should be validated (if 2 are given, 2 different checkpoints of model_best will be kept, one for each validating split).')
parser.add_argument('--log_step', default=50, type=int, help='Every number of steps the log will be printed.')
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image encoder.')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embeddings.')

# Other
parser.add_argument('--gradcam', action='store_true', help='Keep gradients & activations computed while encoding the images to further interprete what the network uses to make its decision.')
parser.add_argument('--studied_split', default="val", help="Split to be used for the computation (this does not impact the usual training & validation pipeline, but impacts other scripts (for evaluation or visualizations purposes)).")


def verify_input_args(args):
	"""
	Check that saving directories exist (or create them).
	Define default values for each dataset.
	"""

	############################################################################
	# --- Check that directories exist (or create them)

	# training ckpt directory
	ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
	if not os.path.isdir(args.ckpt_dir):
		print('Creating a directory: {}'.format(ckpt_dir))
		os.makedirs(ckpt_dir)

	# validated ckpt directories
	args.validate = args.validate.split('-') # splits for validation
	for split in args.validate:
		ckpt_val_dir = os.path.join(ckpt_dir, split)
		if not os.path.isdir(ckpt_val_dir):
			print('Creating a directory: {}'.format(ckpt_val_dir))
			os.makedirs(ckpt_val_dir)

	# prediction directory
	if not os.path.isdir(args.ranking_dir):
		os.makedirs(args.ranking_dir)

	############################################################################
	# --- Process input arguments: deduce some new arguments from provided ones.

	if args.wemb_type == "None":
		args.wemb_type = None

	# Number and name of data categories
	args.name_categories = [None]
	args.recall_k_values = [1, 10, 50]
	args.recall_subset_k_values = None
	args.study_per_category = False # to evaluate the model on each category separately (case of a dataset with multiple categories)
	if args.data_name == 'fashionIQ':
		args.name_categories = ["dress", "shirt", "toptee"]
		args.recall_k_values = [10, 50]
		args.study_per_category = True
	elif args.data_name == 'cirr':
		args.recall_k_values = [1, 5, 10, 50]
		args.recall_subset_k_values = [1, 2, 3]
	args.number_categories = len(args.name_categories)
		
	return args