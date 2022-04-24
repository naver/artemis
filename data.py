import os
import torch
from tqdm import tqdm
import threading
_lock = threading.Lock()

from transforms import get_transform
from dataset_fashionIQ import FashionIQDataset
from dataset_shoes import ShoesDataset
from dataset_cirr import CIRRDataset
from dataset_f200k import Fashion200K


################################################################################
# *** GET FUNCTIONS FOR DATA LOADERS
################################################################################

# Generic functions

def get_loader_single(opt, vocab, split, transform, what_elements="triplet",
						shuffle=True, drop_last=False):

	# Dataset
	if opt.data_name == 'fashionIQ':
		dataset = FashionIQDataset(split, vocab, transform, what_elements,
					opt.load_image_feature, fashion_categories=opt.categories)
	elif opt.data_name == 'fashion200K':
		dataset = Fashion200K(split, vocab, transform, what_elements, opt.load_image_feature)
	elif opt.data_name == 'shoes':
		dataset = ShoesDataset(split, vocab, transform, what_elements, opt.load_image_feature)
	elif opt.data_name == 'cirr':
		dataset = CIRRDataset(split, vocab, transform, what_elements, opt.load_image_feature)

	# Data loader
	collate_fn = get_collate_fn(what_elements)
	data_loader = torch.utils.data.DataLoader(dataset=dataset,
											batch_size=opt.batch_size,
											shuffle=shuffle,
											pin_memory=True,
											num_workers=opt.workers,
											drop_last=drop_last,
											collate_fn=collate_fn)

	print("#### dataset size:", len(dataset))
	print("#### batch size: {} (drop last? {})".format(opt.batch_size, drop_last))

	return data_loader


def get_eval_loader_generic(opt, vocab, split, what_elements):
	transform = get_transform(opt, phase="eval")
	loader = get_loader_single(opt, vocab, split, transform, what_elements,
									shuffle=False, drop_last=False)
	return loader

# Functions to be called from outside this file

def get_train_loader(opt, vocab, split='train', shuffle=True):
	transform = get_transform(opt, phase="train")
	triplet_loader = get_loader_single(opt, vocab, split, transform, "triplet",
											shuffle=shuffle, drop_last=True)
	return triplet_loader


def get_eval_loaders(opt, vocab, split='val'):
	queries_loader = get_eval_loader_generic(opt, vocab, split, "query")
	targets_loader = get_eval_loader_generic(opt, vocab, split, "target")
	return queries_loader, targets_loader


def get_subset_loader(opt, vocab, split='val'):
	return get_eval_loader_generic(opt, vocab, split, "subset")


def get_soft_targets_loader(opt, vocab, split='val'):
	return get_eval_loader_generic(opt, vocab, split, "soft_targets")


################################################################################
# *** COLLATE FN FUNCTIONS
################################################################################

def get_collate_fn(what_elements):

	if what_elements=='triplet':
		collate_fn_func = collate_fn_triplet
	elif what_elements=='query':
		collate_fn_func = collate_fn_query
	elif what_elements=='target':
		collate_fn_func = collate_fn_img_with_id
	elif what_elements=='subset':
		collate_fn_func = collate_fn_tensor_with_index
	elif what_elements == 'soft_targets':
		collate_fn_func = collate_fn_direct

	return collate_fn_func


def collate_fn_triplet(data):
	"""Build mini-batch tensors from a list of tuples.
	Args:
		data: list of (images_src, sentences, images_trg, raw_caption, dataset_index) tuple.
			- images (src/trg): torch tensor of shape (3, 256, 256) or (*).
			- sentence: torch tensor of shape (?); variable length.
			- raw_caption: string
			- dataset_index: integer

	Returns:
		images_src: torch tensor of shape (batch_size, 3, 256, 256)  or (batch_size, *).
		sentences_padded: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded sentence.
		images_trg: same as images_src for target images
		raw_caps: list of the involved captions, as raw text.
		dataset_ids: index of the element in the dataset.
	"""

	# Sort a data list by sentence length
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images_src, sentences, images_trg, raw_caps, dataset_ids = zip(*data)

	# Merge images (convert tuple of 3D tensor to 4D tensor)
	images_src = torch.stack(images_src, 0)
	images_trg = torch.stack(images_trg, 0)

	# Merge sentences (convert tuple of 1D tensor to 2D tensor)
	lengths = torch.tensor([len(cap) for cap in sentences])
	sentences_padded = torch.zeros(len(sentences), max(lengths)).long()
	for i, cap in enumerate(sentences):
		end = lengths[i]
		sentences_padded[i, :end] = cap[:end]

	return images_src, sentences_padded, lengths, images_trg, raw_caps, dataset_ids


def collate_fn_query(data):
	"""Build mini-batch tensors from a list of tuples.
	Args:
		data: list of (image_src, sentence, img_src_id, img_trg_id, raw_caption, dataset_index) tuple.
			- image_src : torch tensor of shape (3, 256, 256) or (*).
			- sentence: torch tensor of shape (?); variable length.
			- img_src_id: integer
			- img_trg_id: list of integers
			- raw_caption: string
			- dataset_index: integer

	Returns:
		images_src: torch tensor of shape (batch_size, 3, 256, 256) or (batch_size, *).
		sentences_padded: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded sentence.
		img_src_ids: list of ids of the source image
		img_trg_ids: list of lists, where each list is a list of target image ids
		raw_caps: list of the involved captions, as raw text.
		dataset_ids: index of the element in the dataset.
	"""

	# Sort a data list by sentence length
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images_src, sentences, img_src_ids, img_trg_ids, raw_caps, dataset_ids = zip(*data)

	# Merge images (convert tuple of 3D tensor to 4D tensor)
	images_src = torch.stack(images_src, 0)

	# Merge sentences (convert tuple of 1D tensor to 2D tensor)
	lengths = torch.tensor([len(cap) for cap in sentences])
	sentences_padded = torch.zeros(len(sentences), max(lengths)).long()
	for i, cap in enumerate(sentences):
		end = lengths[i]
		sentences_padded[i, :end] = cap[:end]

	return images_src, sentences_padded, lengths, img_src_ids, img_trg_ids, raw_caps, dataset_ids


def collate_fn_img_with_id(data):
	"""Build mini-batch tensors from a list of tuples.
	Args:
		data: list of (image, id, dataset_index) tuple.
			- images: torch tensor of shape (3, 256, 256) or (*)
			- id: integer
			- dataset_index: integer

	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256) or (batch_size, *).
		ids: IDs of the image
		dataset_ids: index of the element in the dataset.
	"""
	# Extract data
	images, ids, dataset_ids = zip(*data)
	# Merge images (convert tuple of 3D tensor to 4D tensor)
	images = torch.stack(images, 0)
	return images, ids, dataset_ids


def collate_fn_tensor_with_index(data):
	"""Build mini-batch tensors from a list of tuples.
	Args:
		data: list of (thing, dataset_index) tuple.
			- thing: torch tensor of shape (*).
			- dataset_index: integer

	Returns:
		things: torch tensor of shape (batch_size, *).
		dataset_ids: index of the element in the dataset.
	"""
	# Extract data
	things, dataset_ids = zip(*data)
	# Merge things (convert tuple of *D tensor to (*+1)D tensor)
	things = torch.stack(things, 0)
	return things, dataset_ids


def collate_fn_direct(data):
	"""Build mini-batch tensors from a list of tuples.
	Args:
		data: list of tuples

	Returns:
		one list per element in a tuple, gathering all the elements at the
		same position in the tuple, accross all tuples
	"""
	return zip(*data)