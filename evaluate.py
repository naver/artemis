import os
import time
from tqdm import tqdm
import pickle
import copy
import torch

from option import parser, verify_input_args
import data
from vocab import Vocabulary
from artemis_model import ARTEMIS
from tirg_model import TIRG
from evaluate_cirr import deal_with_CIRR

# Included functions:
# - validate
# - compute_and_process_compatibility_scores
# - compute_necessary_embeddings_img
# - get_rank_of_GT
# - get_recall
# - results_func
# - load_model
# - main

def validate(model, args, vocab, output_type="metrics", max_retrieve = 50, split='val'):
	"""
	Input:
		model, args, vocab;
		output_type: either "metrics" or "rankings",
		max_retrieve: top number of propositions to keep for a given query,
		split;

	Output:
	- if output_type is "metrics": returns a message presenting the results and
	    a validation score. If applicable, results are presented for each data
	    category.
	- if output_type is "rankings": tensor of size (#queries, max_retrieved)
	    containing the top ranked target ids corresponding to each query. If
	    applicable, results are organized per data category.
	"""

	# Special case for CIRR: metrics are computed at the end, based on the rankings
	output_type_inpractice = "rankings" if args.data_name == "cirr" else output_type

	# Initializations
	results = []
	categories = args.name_categories if ("all" in args.categories) else args.categories.split(' ') # if not applicable, `categories` becomes `[None]``

	# Switch to eval mode
	model.eval()

	# Compute measures or rankings
	for category in categories:

		# specify the category to be studied, if applicable
		opt = copy.deepcopy(args)
		if args.study_per_category and (args.number_categories > 1):
			opt.categories = category

		# load data
		queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, split)

		# compute & process compatibility scores 
		with torch.no_grad(): # no need to retain the computational graph and gradients
			start = time.time()
			res = compute_and_process_compatibility_scores(queries_loader, targets_loader,
													model, opt, output_type_inpractice,
													max_retrieve)
			end = time.time()
			print("\nProcessing time : ", end - start)

		# store results for presentation / further process
		results.append(res)

	if output_type=="metrics":
		# compute additional metrics and present properly the results
		if args.data_name == "cirr":
			# also compute the subset ranking
			message, val_mes = deal_with_CIRR(args, vocab, results[0], 	split) # [0] because no category in CIRR 
		else:
			message, val_mes = results_func(results, args)
		return message, val_mes

	return results


def compute_and_process_compatibility_scores(data_loader_query, data_loader_target,
										model, args, output_type="metrics",
										max_retrieve=50):
	"""
	Compute the compatibility score of each query of the query dataloader with
	regard to all the candidate targets of the target dataloader, and process it.
	To save some memory at evaluation time, this function should be called "with
	torch.no_grad()".

	Input:
		output_type: either "metrics" or "rankings"

	Output:
	- if output_type is "metrics": tensor of size (#queries) containing the rank
		of the best ranked correct target for each query;
	- if output_type is "rankings": tensor of size (#queries, max_retrieved)
	  	containing the top ranked target ids corresponding to each query.
	"""

	nb_queries= len(data_loader_query.dataset)

	# Initialize output
	if output_type=="metrics":
		# return the rank of the best ranked correct target
		ret = torch.zeros(nb_queries, requires_grad=False)
	else:
		# return the top propositions for each query
		ret = torch.zeros(nb_queries, max_retrieve, requires_grad=False).int()

	# Pre-compute image embeddings (includes all target & reference images)
	all_img_embs = compute_necessary_embeddings_img(data_loader_target, model, args)

	# Compute and process compatibility scores (process by batch)
	for data in tqdm(data_loader_query):

		# Get query data
		_, txt, txt_len, img_src_ids, img_trg_ids, _, indices = data
		if torch.cuda.is_available():
			txt, txt_len = txt.cuda(), txt_len.cuda()

		# Compute query embeddings for the whole batch
		# (the reference image embedding is included in `all_img_embs`, so there
		# is only the text embedding left to compute)
		txt_embs = model.get_txt_embedding(txt, txt_len)

		# Process each query of the batch one by one
		for i, index in enumerate(indices):

			# Select data related to the current query
			txt_emb = txt_embs[i]
			img_src_id = img_src_ids[i]
			GT_indices = img_trg_ids[i]
			img_src_emb = all_img_embs[img_src_id]

			# Compute compatibility scores between the query and each candidate target
			cs = model.get_compatibility_from_embeddings_one_query_multiple_targets(
										img_src_emb, txt_emb, all_img_embs)

			# Remove the source image from the ranking
			cs[img_src_id] = float('-inf')

			# Rank targets
			cs_sorted_ind = cs.sort(descending=True)[1]
			
			# Store results
			if output_type == "metrics":
				ret[index] = get_rank_of_GT(cs_sorted_ind, GT_indices)[0]
			else:
				ret[index, :max_retrieve] = cs_sorted_ind[:max_retrieve].cpu().int()

	return ret


def compute_necessary_embeddings_img(data_loader_target, model, args):

	"""
	Compute the embeddings of the target images.
	To save some memory, this function should be called "with torch.no_grad()".

	Input:
		data_loader_target: dataloader providing images and indices of the provided
			items within the dataloader
		model, args;

	Output:
		img_trg_embs (cuda)
	"""

	img_trg_embs = None

	for data in tqdm(data_loader_target):

		# Get target data
		img_trg, _, indices = data
		indices = torch.tensor(indices)
		if torch.cuda.is_available():
			img_trg = img_trg.cuda()

		# Compute embedding
		img_trg_emb = model.get_image_embedding(img_trg)

		# Initialize the output embeddings if not done already
		if img_trg_embs is None:
			emb_sz = [len(data_loader_target.dataset), args.embed_dim]
			img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False)
			if torch.cuda.is_available():
				img_trg_embs = img_trg_embs.cuda()

		# Preserve the embeddings by copying them
		if torch.cuda.is_available():
			img_trg_embs[indices] = img_trg_emb
		else :
			img_trg_embs[indices] = img_trg_emb.cpu()

	return img_trg_embs


def get_rank_of_GT(sorted_ind, GT_indices):
	"""
	Get the rank of the best ranked correct target provided the target ranking
	(targets are identified by indices). Given two acceptable correct targets of
	respective indices x and y, if the target of index x has a better rank than
	the target of index y, then the returned value for `rank_of_GT ` is the rank
	of the target of index x, and the value of `best_GT` is x.

	Input:
		sorted_ind: tensor of size (number of candidate targets), containing the
			candidate target indices sorted in decreasing order of relevance with
			regard to a given query.
		GT_indices: list of correct target indices for a given query.

	Output:
		rank_of_GT: rank of the best ranked correct target, if it is found
			(+inf is returned otherwise)
		best_GT: index of the best ranked correct target

	"""
	rank_of_GT = float('+inf')
	best_GT = None
	for GT_index in GT_indices:
		tmp = torch.nonzero(sorted_ind == GT_index)
		if tmp.size(0) > 0: # the GT_index was found in the ranking
			tmp = tmp.item()
			if tmp < rank_of_GT:
				rank_of_GT = tmp
				best_GT = GT_index
	return rank_of_GT, best_GT


def get_recall(rank_of_GT, K):
	return 100 * (rank_of_GT < K).float().mean()


def results_func(results, args):
	"""
	Compute metrics over the dataset and present them properly.
	The result presentation and the computation of the metric might depend
	on particular options/arguments (use the `args`).

	Input:
	    results: list containing one tensor per data category (or just one
	        tensor if the dataset has no particular categories). The tensor is
	        of size (number of queries) and ontains the rank of the best ranked
	        correct target.
		args: argument parser from option.py

	Ouput:
		message: string message to print or to log
		val_mes: measure to monitor validation (early stopping...)
	"""

	nb_categories = len(results)

	# --- Initialize a dictionary to hold the results to present
	H = {"r%d"%k:[] for k in args.recall_k_values}
	H.update({"medr":[], "meanr":[], "nb_queries":[]})

	# --- Iterate over categories
	for i in range(nb_categories):
		# get measures about the rank of the best ranked correct target
		# for category i
		for k in args.recall_k_values:
			H["r%d"%k].append(get_recall(results[i], k))
		H["medr"].append(torch.floor(torch.median(results[i])) + 1)
		H["meanr"].append(results[i].mean() + 1)
		H["nb_queries"].append(len(results[i]))

	# --- Rearrange results (aggregate category-specific results)
	H["avg_per_cat"] = [sum([H["r%d"%k][i] for k in args.recall_k_values])/len(args.recall_k_values) for i in range(nb_categories)]
	val_mes = sum(H["avg_per_cat"])/nb_categories
	H["nb_total_queries"] = sum(H["nb_queries"])
	for k in args.recall_k_values:
		H["R%d"%k] = sum([H["r%d"%k][i]*H["nb_queries"][i] for i in range(nb_categories)])/H["nb_total_queries"]
	H["rsum"] = sum([H["R%d"%k] for k in args.recall_k_values])
	H["med_rsum"] = sum(H["medr"])
	H["mean_rsum"] = sum(H["meanr"])

	# --- Present the results of H in a single string message
	message = ""

	# multiple-category case: print category-specific results
	if nb_categories > 1:
		categories = args.name_categories if ("all" in args.categories) else args.categories
		cat_detail = ", ".join(["%.2f ({})".format(cat) for cat in categories])

		message += ("\nMedian rank: " + cat_detail) % tuple(H["medr"])
		message += ("\nMean rank: " + cat_detail) % tuple(H["meanr"])
		for k in args.recall_k_values:
			message += ("\nMetric R@%d: " + cat_detail) \
						% tuple([k]+H["r%d"%k])

		# for each category, average recall metrics over the different k values
		message += ("\nRecall average: " + cat_detail) % tuple(H["avg_per_cat"])

		# for each k value, average recall metrics over categories
		# (remove the normalization per the number of queries)
		message += "\nGlobal recall metrics: {}".format( \
						", ".join(["%.2f (R@%d)" % (H["R%d"%k], k) \
						for k in args.recall_k_values]))

	# single category case
	else:
		message += "\nMedian rank: %.2f" % (H["medr"][0])
		message += "\nMean rank: %.2f" % (H["meanr"][0])
		for k in args.recall_k_values:
			message += "\nMetric R@%d: %.2f" % (k, H["r%d"%k][0])

	message += "\nValidation measure: %.2f\n" % (val_mes)

	return message, val_mes


def load_model(args):

	# Load vocabulary
	vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
	assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
	vocab = pickle.load(open(vocab_path, 'rb'))

	# Setup model
	if args.model_version == "TIRG":
		model = TIRG(vocab.word2idx, args)
	else:
		# model version is ARTEMIS or one of its ablatives
		model = ARTEMIS(vocab.word2idx, args)
	print("Model version:", args.model_version)

	if torch.cuda.is_available():
		model = model.cuda()
		torch.backends.cudnn.benchmark = True

	# Load model weights
	if args.ckpt:

		# load checkpoint
		assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
		print(f"Loading file {args.ckpt}.")

		if torch.cuda.is_available():
			model.load_state_dict(torch.load(args.ckpt)['model'])
		else :
			state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
			model.load_state_dict(state_dict)
		print("Model: resume from provided state.")

	return args, model, vocab


if __name__ == '__main__':

	args = verify_input_args(parser.parse_args())

	# Load model & vocab
	args, model, vocab = load_model(args)

	start = time.time()
	with torch.no_grad():
		message, _ = validate(model, args, vocab, split = args.studied_split)
	print(message)

	# save printed message on .txt file
	basename = ""
	if os.path.basename(args.ckpt) != "model_best.pth":
		basename = "_%s" % os.path.basename(os.path.basename(args.ckpt)).split(".")[0]
	if args.data_name == "fashionIQ":
		save_txt = os.path.abspath( os.path.join(args.ckpt, os.path.pardir, os.path.pardir, 'eval_message%s.txt' % basename) )
	else:
		save_txt = os.path.abspath( os.path.join(args.ckpt, os.path.pardir, 'eval_message%s.txt' % basename) )
	with open(save_txt, 'a') as f:
		f.write(args.data_name + ' ' + args.studied_split + ' ' + args.exp_name + '\n######')
		f.write(message + '\n######\n')

	end = time.time()
	print("\nProcessing time : ", end - start)
