import data
import json
import os

def deal_with_CIRR(args, vocab, cs_sorted_ind, split):
	"""
	Depending on the split:
	- val split: annotations are available, one can output metrics.
	- test split: no annotation available, one needs to produce json files and
		submit them to the testing server.

		More information at:
		https://github.com/Cuberick-Orion/CIRR/blob/main/Test-split_server.md
		
		Code inspired from:
		https://github.com/Cuberick-Orion/CIRPLANT/blob/fd989870194910fad05c57aec8e361ee15587219/model/OSCAR/OSCAR_CIRPLANT.py#L346

	"""

	# Setup
	cirr_subset_dataloader = data.get_subset_loader(args, vocab, split)
	subset_ids = get_data_in_list(cirr_subset_dataloader)

	# Evaluation
	if split == "val":

		soft_targets = get_data_in_list(data.get_soft_targets_loader(args, vocab, split))
		rK, rsubK = eval_for_CIRR(cs_sorted_ind, subset_ids, soft_targets, args.recall_k_values, args.recall_subset_k_values)
	
		message = "\n\n>> EVALUATION <<"
		message += results2string(rK, "R")
		message += results2string(rsubK, "R_sub")
		val_mes = 0.5*(rK[1][1] + rsubK[0][1])
		message += "\n(R@%d + R_sub@%d)/2: %.2f" % (rK[1][0], rsubK[0][0], val_mes)

		message += "\nValidation measure: %.2f\n" % (val_mes)

		return message, val_mes

	elif split == "test":

		dir_save = os.path.join(args.ranking_dir, args.exp_name)
		if not os.path.isdir(dir_save):
			os.makedirs(dir_save)

		# get info to produce prediction files
		image_id2name = cirr_subset_dataloader.dataset.image_id2name
		pair_ids = cirr_subset_dataloader.dataset.get_pair_ids()
		assert len(pair_ids) == cs_sorted_ind.size(0), "Size mismatch: studying {} queries, but got results for {} of them.".format(len(pair_ids), cs_sorted_ind.size(0))

		# prediction file: recall
		d = {pair_ids[i]:[image_id2name[ind] for ind in cs_sorted_ind[i]][:50] for i in range(len(pair_ids))}
		d.update({"version":"rc2", "metric":"recall"})
		message = dict_to_json(d, os.path.join(dir_save, "cirr_recall.json")) + "\n"

		# prediction file: recall subset
		d = {pair_ids[i]:[image_id2name[ind] for ind in \
							[ind_subset for ind_subset in cs_sorted_ind[i] \
							if ind_subset in subset_ids[i]]][:3] \
							for i in range(len(pair_ids))}
		d.update({"version":"rc2", "metric":"recall_subset"})
		message += dict_to_json(d, os.path.join(dir_save, "cirr_recall_subset.json"))
		
		return message, 1


def results2string(values, metric_name):
	message = ""
	for k, v in values:
		message += ("\nMetric {}@%d: %.2f" % (k, v)).format(metric_name)
	return message


def get_data_in_list(data_loader):
	L = []
	for data in data_loader:
		d, _ = data # skip the dataset index ; is already a list (batch size)
		if type(d) is tuple:
			L += list(d)
		else: # torch.tensor
			L += d.tolist()
	return L


def eval_for_CIRR(cs_sorted_ind, subset_ids, soft_targets,
					recall_k_values, recall_subset_k_values):

	"""
	Input:
		cs_sorted_ind: torch.tensor size (nb of queries, retrieved nb of candidate
			targets), containing the indices of the top ranked candidate targets
			for each query.
		subset_ids: list of size (nb of queries). Its sublists contain the
			indices of a subset of potential targets to compare to, for each query,
			as a specific metric, R_subset@K.
		soft_targets: list of size (nb of queries). It contains one dictionary
			for each query, such that {soft target id: soft target qualification},
			indicating what candidate target (soft target id) can be considered
			as acceptable target (soft target qualification: 1 is OK, 0.5 is 50%
			OK, and -1 is not OK) for the given query.

	More info on the CIRR Github project page:
	https://github.com/Cuberick-Orion/CIRR
	
	"""

	out_rK = []
	# Recall@K
	for k in recall_k_values:
		r = 0.0
		for i, nns in enumerate(cs_sorted_ind):
			highest_r = 0.0
			for ii,ss in soft_targets[i].items():
				if ii in nns[:k]:
					highest_r = max(highest_r, ss) # update the score
			r += highest_r
		r /= len(cs_sorted_ind)
		out_rK += [(k, r*100)]
	
	# Recall_subset@K
	out_rsubK = []
	for k in recall_subset_k_values:
		r = 0.0
		for i, nns in enumerate(cs_sorted_ind):
			nns = [iii for iii in nns if iii in subset_ids[i]]
			highest_r = 0.0
			for ii,ss in soft_targets[i].items():
				if ii in nns[:k]:
					highest_r = max(highest_r, ss)
			r += highest_r
		r /= len(cs_sorted_ind)
		out_rsubK += [(k,r*100)]

	return out_rK, out_rsubK


def dict_to_json(d, filepath):
	with open(filepath, "w") as file:
		json.dump(d, file)
	return "File saved at {}".format(filepath)