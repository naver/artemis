#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
"""
This script enables to produce heatmaps for the EM & IS scores of the ARTEMIS
model.

Change the global parameters below to precise what / how many heatmaps should be
generated, and run this script with the same arguments as when evaluating a
model, with `--gradcam` in addition.
"""

import os 
import cv2
import numpy as np
import copy
import json

import torch
from torch.autograd import grad

import data
from vocab import Vocabulary
from utils import params_require_grad
from artemis_model import ARTEMIS
from evaluate import load_model, compute_and_process_compatibility_scores
from option import parser, verify_input_args

################################################################################
# *** GLOBAL PARAMETERS
################################################################################

# whether to generate heatmaps for the queries yielding the best results (the
# ground truth target image is well ranked)
ONLY_BEST_RESULTS = True

# number of queries to study
NUMBER_OF_EXAMPLES = 5

# number of coefficients contributing the most to a given score, that should be
# considered for backpropagation, in the GradCAM algorithm. If the score is
# computed as the dot product of two vectors `a` and `b`, the coefficients are
# given by the point-wise product of `a` and `b`.
NUMBER_OF_MAIN_COEFF = 3

################################################################################
# *** GENERATE & SAVE HEATMAPS
################################################################################

def main_generate_heatmaps(args, model, vocab):
	"""
	Potentially find the indices of the most relevant data examples (i.e. data
	examples whose expected target image is well ranked by the model), and
	generate heatmaps for them.
	"""

	categories = args.name_categories if ("all" in args.categories) else args.categories # if not applicable, `categories` becomes `[None]`

	for category in categories:

		# -- Find the indices of relevant data to use for heatmaps generation
		if ONLY_BEST_RESULTS:

			# Specify the category to be studied, if applicable
			opt = copy.deepcopy(args)
			if args.study_per_category and (args.number_categories > 1):
				opt.categories = category

			# Load data
			queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, args.studied_split)

			# Find the best triplets
			studied_indices, rank_of_GT = find_best_results(model, opt, queries_loader, targets_loader)

			# Save metadata
			d = {studied_indices[i]: int(rank_of_GT[i]) for i in range(len(studied_indices))}
			directory = os.path.join(args.heatmap_dir, args.exp_name)
			if not os.path.isdir(directory):
				os.makedirs(directory)
			with open(os.path.join(directory, "metadata.json"), "a") as f:
				f.write(f"\n\nCategory: {category} \n")
				json.dump(d, f)
			print(f"Saving metadata (studied data indices, rank of GT) at {os.path.join(directory, 'metadata.json')}.")

		else:
			studied_indices = None

		# -- Generate heatmaps

		# Specify the category to be studied, if applicable
		opt = copy.deepcopy(args)
		if args.study_per_category and (args.number_categories > 1):
			opt.categories = category

		# Load data
		opt.batch_size = 1
		triplet_loader = data.get_train_loader(opt, vocab, split=args.studied_split, shuffle=False)

		# Generate heatmaps
		generate_heatmaps_from_dataloader(triplet_loader, model, opt,
								studied_indices=studied_indices)


def find_best_results(model, opt, queries_loader, targets_loader):
	"""
	Return:
	- a list containing the indices (within the dataloader) of the queries
		raising the best results (ie. the ground truth target is well ranked)
	- a list containing the rank of the correct target for the selected queries
	"""

	# Switch to eval mode
	model.eval()

	# Rank all the potential targets for all the queries
	with torch.no_grad(): # no need to retain the computational graph and gradients
		rank_of_GT = compute_and_process_compatibility_scores(queries_loader, targets_loader,
												model, opt, output_type="metrics")

	# Select the queries whose expected target is ranked the best
	data_ids = rank_of_GT.sort()[1][:NUMBER_OF_EXAMPLES]	

	return data_ids.tolist(), rank_of_GT[data_ids].tolist()


def generate_heatmaps_from_dataloader(data_loader, model, args,
							studied_indices=None):
	"""
	Generate and save heatmaps for several (specific) data examples from the
	provided dataloader.

	Input:
	    data_loader: train type, must handle batchs of size 1.
	    studied_indices: indices of the data examples that should be studied,
	        within the dataloader. If None, the processed data examples are
	        taken in the order of the provided dataloader.
	"""

	# set the evaluation mode
	model.eval()
	params_require_grad(model.txt_enc, False)
	
	# iterate over the dataloader to produce the heatmaps
	data_loader_iterator, itr = iter(data_loader), 0
	while itr < NUMBER_OF_EXAMPLES:

		# Get data
		img_src, txt, txt_len, img_trg, ret_caps, data_id = next(data_loader_iterator)
		example_number = data_id[0] # batch size is 1

		if (studied_indices is None) or (example_number in studied_indices):

			# Process data
			generate_heatmap_from_single_data(args, model, img_src, txt, txt_len,
												img_trg, example_number)

			# store image identifiers & caption
			formated_caption, ref_identifier, trg_identifier = data_loader.dataset.get_triplet_info(example_number)
			directory = os.path.join(args.heatmap_dir, args.exp_name, str(example_number))
			with open(os.path.join(directory, "metadata.txt"), "a") as f:
				f.write(f"{example_number}*{formated_caption}*{ref_identifier}*{trg_identifier}\n")

			# Clean
			del img_src, txt, txt_len, img_trg, ret_caps, data_id

			# Iterate
			itr += 1


def generate_heatmap_from_single_data(args, model, img_src, txt, txt_len,
										img_trg, example_number):
	"""
	Generate and save heatmaps for a given data example.

	Input:
		example_number: index of the current data example in a dataloader.
	"""

	if torch.cuda.is_available():
		img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()

	img_src = img_src.requires_grad_(True)
	img_trg = img_trg.requires_grad_(True)

	# Forward pass, during which intermediate results are stored in model.hold_results
	_ = model.forward_save_intermediary(img_src, img_trg, txt, txt_len) # output scores

	# Generate heatmaps for each score, EM & IS
	heatmap_from_score(args, model, example_number,
						"IS", "A_IS_r", "A_IS_t", img_trg,
						img_src=img_src, r_is_involved=True)
	heatmap_from_score(args, model, example_number,
						"EM", "Tr_m", "A_EM_t", img_trg)


def heatmap_from_score(args, model, example_number, s_name,
							query_contrib_name, t_contrib_name,
							img_trg, img_src=None, r_is_involved=False):
	"""
	Generate and save heatmaps for a given data example and score.
	The studied score (indicated by s_name) results from the dot product of two
	known subresults, one that is query-related (query_contrib_name), and the
	other that is target-related (t_contrib_name).

	Input:
		example_number: index of the current data example in a dataloader.
		s_name: score name (EM|IS)
		query_contrib_name: name of the query-related subresult that contributes
			to the score to study (Tr_m|A_IS_r)
		t_contrib_name: name of the target-related subresult that contributes to
			the score to study (A_EM_t|A_IS_t)
		r_is_involved: whether the reference image is involved in the score.
	"""

	r_heatmap_tmp = None, None
	t_heatmap_tmp = None, None

	# get images activations
	if r_is_involved:
		r_activation = model.hold_results["r_activation"] # size (batch_size, channels, 7, 7)
	t_activation = model.hold_results["t_activation"] # size (batch_size, channels, 7, 7)

	# find the coeffs that contribute the most to the score
	query_contrib = model.hold_results[query_contrib_name]
	t_contrib = model.hold_results[t_contrib_name]
	main_coeffs = get_main_coeffs(query_contrib, t_contrib)

	# produce one heatmap per main coeff for each score
	for main_coeff in main_coeffs:
		
		# extract the contribution of the selected output coeff
		score_contrib = (query_contrib*t_contrib)[:,main_coeff]

		# get pooled gradients across the channels for the given output coeff
		if r_is_involved:
			r_weights = get_weights(model, score_contrib, r_activation) # size (batch_size, channels)
		t_weights = get_weights(model, score_contrib, t_activation) # size (batch_size, channels)

		# weight the channels of the activation map with the pooled gradients,
		# and add this weighted activation map to the total heatmap, accounting for
		# the contribution of the selected output coeff, if only one heatmap is required ;
		# otherwise, save the current heatmap
		if r_is_involved:
			r_heatmap_tmp = (r_activation * r_weights.view(args.batch_size, -1, 1, 1)).sum(dim=1).detach().cpu() # size (batch_size, 7, 7)
		t_heatmap_tmp = (t_activation * t_weights.view(args.batch_size, -1, 1, 1)).sum(dim=1).detach().cpu() # size (batch_size, 7, 7)

		save_heatmaps(args,
						example_number,
						f"{s_name}_coeff_{main_coeff}",
						round(score_contrib[0].item(), 4),
						t_heatmap_tmp,
						img_trg,
						r_heatmap_tmp,
						img_src,
						r_is_involved)

	if r_is_involved:
		del r_activation, r_weights, r_heatmap_tmp
	del t_activation, t_weights, score_contrib, t_heatmap_tmp
	del query_contrib, t_contrib, main_coeffs


def save_heatmaps(args, example_number, s_name, s_value, t_heatmap, img_trg,
					r_heatmap=None, img_src=None, r_is_involved=False):
	"""
	Save provided heatmaps.
	Files are stored in {args.ranking_dir}/{prefix}/{example_number}/, with the
	names {s_name}_on_trg_heatmap.jpg and {s_name}_on_src_heatmap.jpg.
	Additional information is stored as metadata.

	Input:
		args: parsed arguments
		example_number: index of the current data example in a dataloader.
		s_name: score name (or alternatively: what is observed)
		s_value: score value (or alternatively: metric about what is observed)
		t_heatmap: heatmap to apply on the target image.
		img_trg: target image, as it is processed by the model (cropped & resized...)
		r_heatmap: heatmap to apply on the reference image (optional)
		img_src: reference image, as it is processed by the model (cropped & resized...)
		r_is_involved: whether an heatmap and an image is provided for the reference image.
	"""

	# normalize the heatmaps
	if r_is_involved:
		r_heatmap = normalize_heatmap(r_heatmap)
	t_heatmap = normalize_heatmap(t_heatmap)

	# store interpretations
	directory = os.path.join(args.heatmap_dir, args.exp_name, str(example_number))
	if not os.path.isdir(directory):
		os.makedirs(directory)

	filename = os.path.join(directory, '{}_heatmap.jpg')
	if r_is_involved:
		merge_heatmap_on_image(r_heatmap, img_src, filename.format(f"{s_name}_on_src"))
	merge_heatmap_on_image(t_heatmap, img_trg, filename.format(f"{s_name}_on_trg"))

	# store captions as well for later visualization
	with open(os.path.join(directory, "metadata.txt"), "a") as f:
		f.write(f"{example_number}*{s_name}*{s_value}\n")


def get_weights(model, output, conv_activation):
	"""
	For a given input, the model produces a corresponding activation map
	(`conv_activation`) and value (`output`). 
	This method returns the gradients of the value with regard to the activation
	map, pooled over the activation weights.
	"""
	# reset the gradient for a fresh backward pass
	model.zero_grad()
	# backward pass : get the gradient of the output with respect to the last convolutional layer
	gradients = grad(outputs=output, inputs=conv_activation, retain_graph=True)[0] # size (batch_size, 2048 or 512, 7, 7) : batch size, channels, 2D feature map (activation weights)
	# pool the gradients across the channels
	weights = gradients.mean(dim=[2,3]) # size (batch_size, 2048 or 512)
	return weights


def normalize_heatmap(heatmap):
	heatmap = torch.clamp(heatmap, 0) # relu on top of the heatmap
	heatmap = normalize_image(heatmap)
	return heatmap


def normalize_image(img):
	if isinstance(img, torch.Tensor):
		img -= torch.min(img)
		maxi_value = torch.max(img)
	if isinstance(img, np.ndarray):
		img -= np.min(img)
		maxi_value = np.max(img)
	img /= maxi_value if maxi_value > 0 else 0.00001
	return img


def merge_heatmap_on_image(heatmap, initial_img, produced_img_path):
	"""
	Superimpose the heatmap on the initial image.
	The initial image must be the processed version (correctly resized &
	centered/crop) of the original image (since the model made the computation
	on the processed image)

	Input : 
		heatmap: torch tensor of size (1, 7, 7), with values between 0 and 1
		initial_img: (cuda) torch tensor of size (1, 3, 224, 224) -
			unknown range of values
	    produced_img_path: where to save the new image, consisting in the
	    	heatmap superimposed on the initial image
	"""

	# consider the first and unique element of the batch
	# the initial image must be of size (heigth, width, color channels) to fit
	# the cv2 processing (hence the permutation)
	heatmap = heatmap[0].data.numpy()
	initial_img = normalize_image(np.float32(initial_img.cpu()[0].permute(1, 2, 0).data.numpy()))

	# from now :
	#	- heatmap : numpy array with values between 0 and 1
	#	- initial_img : numpy array with values between 0 and 1

	heatmap = cv2.resize(heatmap, (initial_img.shape[0], initial_img.shape[1])) # interpolation of the heatmap
	heatmap = np.uint8(255 * heatmap) # convert between 0 and 255
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # infer the heatmap colorization
	heatmap = np.float32(heatmap) / 255. # convert between 0 and 1
	superimposed_img = heatmap + initial_img # merge heatmap & initial image
	superimposed_img = normalize_image(superimposed_img) # normalize the produced image
	superimposed_img = np.uint8(255 * superimposed_img) # convert between 0 and 255
	cv2.imwrite(produced_img_path, superimposed_img)
	print("Interpretable image registred at : {}".format(produced_img_path))


def get_main_coeffs(term_1, term_2):
	"""
	Input:
		term_1, term_2: vectors

	Output:
		indices of the coefficients that contribute the most to the score
		resulting from the dot product of `term_1` and `term_2`.
	"""
	return (term_1 * term_2).detach().cpu()[0].sort()[1][-NUMBER_OF_MAIN_COEFF:] # [0] because we consider the first (and unique) element of the batch


################################################################################
#### MAIN


if __name__ == '__main__':

	args = verify_input_args(parser.parse_args())

	# Load model
	args, model, vocab = load_model(args)

	# Generate heatmaps 
	main_generate_heatmaps(args, model, vocab)