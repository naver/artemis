#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import itertools
import json as jsonmod

from dataset import MyDataset
from config import FASHIONIQ_IMAGE_DIR, FASHIONIQ_ANNOTATION_DIR

class FashionIQDataset(MyDataset):
	"""
	FashionIQ dataset, introduced in "Fashion IQ: A new dataset towards
	retrieving images by natural language feedback"; Hui Wu, Yupeng Gao,
	Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio
	Feris; Proceedings of CVPR, pp. 11307–11317, 2021.
	"""

	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0,
					fashion_categories='all', ** kw):
		"""
		Args:
			fashion_categories: fashion_categories to consider. Expected to be a string such as : "dress toptee".
		"""
		MyDataset.__init__(self, split, FASHIONIQ_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

		fashion_categories = ['dress', 'shirt', 'toptee'] if fashion_categories=='all' else sorted(fashion_categories.split())

		# concatenate in one list the image identifiers of the fashion categories to consider
		image_id2name_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'image_splits', f'split.{fc}.{split}.json') for fc in fashion_categories]
		image_id2name = [self.load_file(a) for a in image_id2name_files]
		self.image_id2name = list(itertools.chain.from_iterable(image_id2name))

		# if necessary, load triplet annotations of the fashion categories to consider
		if self.what_elements in ["query", "triplet"]:
			prefix = 'pair2cap' if split=='test' else 'cap'
			annotations_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'captions', f'{prefix}.{fc}.{split}.json') for fc in fashion_categories]
			annotations = [self.load_file(a) for a in annotations_files]
			self.annotations = list(itertools.chain.from_iterable(annotations))


	def __len__(self):
		if self.what_elements=='target':
			return len(self.image_id2name)
		return 2*len(self.annotations) # 1 annotation = 2 captions = 2 queries/triplets


	def load_file(self, f):
		"""
		Depending on the file, returns:
			- a list of dictionaries with the following format:
				{'target': 'B001AS562I', 'candidate': 'B0088WRQVS', 'captions': ['i taank top', 'has spaghetti straps']}
			- a list of image identifiers
		"""
		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann


	############################################################################
	# *** GET ITEM METHODS
	############################################################################

	def get_triplet(self, idx):

		# NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
		# of each reference-target pair separately, doubling the number of
		# queries
		index = idx // 2 # get the annotation index
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)

		# get data
		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".jpg"
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['target'] + ".jpg"
		img_trg = self.get_transformed_image(path_trg)

		return img_src, text, img_trg, real_text, idx


	def get_query(self, idx):

		# NOTE: following CoSMo (Lee et. al, 2021), we consider the two captions
		# of each reference-target pair separately, doubling the number of
		# queries
		index = idx // 2 # get the annotation index
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)

		# get data
		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".jpg"
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['candidate'])

		img_trg_id = [self.image_id2name.index(ann['target'])]

		return img_src, text, img_src_id, img_trg_id, real_text, idx


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index] + ".jpg"
		img = self.get_transformed_image(path_img)

		return img, img_id, index


	############################################################################
	# *** FORMATTING INFORMATION FOR VISUALIZATION PURPOSES
	############################################################################

	def get_triplet_info(self, idx):
		"""
		Should return 3 strings:
			- the text modifier
			- an identification code (name, relative path...) for the reference image
			- an identification code (name, relative path...) for the target image
		"""
		index = idx // 2
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)
		ann = self.annotations[index]
		return " [and] ".join(ann["captions"][cap_slice]), ann["candidate"], ann["target"]