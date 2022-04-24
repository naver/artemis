#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import itertools
import json as jsonmod
import torch

from dataset import MyDataset
from config import SHOES_IMAGE_DIR, SHOES_ANNOTATION_DIR


class ShoesDataset(MyDataset):
	"""
    Shoes dataset, introduced with "Dialog-based interactive image retrieval";
    Xiaoxiao Guo, Hui Wu, Yu Cheng, Steven Rennie, Gerald Tesauro, and Rogerio
    Feris; Proceedings of NeurIPS, pp. 676–686, 2018.

	Images are extracted from "Automatic attribute discovery and
	characterization from noisy web data"; Tamara L Berg, Alexander C Berg, and
	Jonathan Shih; Proc. ECCV, pp. 663–676, 2010.
	"""

	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):
		"""
		Args:
			split: either "train", "val" : used to know if to do text augmentation
			vocab: vocabulary wrapper.
			transform: tuple (transformer for image, opt.img_transform_version)
			what_elements: specifies what to provide when iterating over the dataset (queries, targets ?)
			load_image_feature: whether to load raw images (if 0, default) or pretrained image feature (if > 0, size of the feature)
		"""

		MyDataset.__init__(self, split, SHOES_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

		# load the paths of the images involved in the split
		self.image_id2name = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'split.{split}.json'))

		# if necessary, load triplet annotations
		if self.what_elements in ["query", "triplet"]:
			self.annotations = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'triplet.{split}.json'))


	def __len__(self):
		if self.what_elements=='target':
			return len(self.image_id2name)
		return len(self.annotations)


	def load_file(self, f):
		"""
		Depending on the file, returns:
			- a list of dictionaries with the following format:
				{'ImageName': '{path_from_{img_dir}}/img_womens_clogs_851.jpg', 'ReferenceImageName': '{path_from_{img_dir}}/img_womens_clogs_512.jpg', 'RelativeCaption': 'is more of a textured material'}
			- a list of image identifiers/paths
		"""
		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann


	############################################################################
	# *** GET ITEM METHODS
	############################################################################

	def get_triplet(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['ImageName']
		img_trg = self.get_transformed_image(path_trg)

		return img_src, text, img_trg, real_text, index


	def get_query(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['ReferenceImageName'])

		img_trg_id = [self.image_id2name.index(ann['ImageName'])]

		return img_src, text, img_src_id, img_trg_id, real_text, index


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index]
		img = self.get_transformed_image(path_img)

		return img, img_id, index


	############################################################################
	# *** FORMATTING INFORMATION FOR VISUALIZATION PURPOSES
	############################################################################

	def get_triplet_info(self, index):
		"""
		Should return 3 strings:
			- the text modifier
			- an identification code (name, relative path...) for the reference image
			- an identification code (name, relative path...) for the target image
		"""
		ann = self.annotations[index]
		return ann["RelativeCaption"], ann["ReferenceImageName"], ann["ImageName"]