import os
import random
import numpy as np

from dataset import MyDataset
from tqdm import tqdm
from config import FASHION200K_IMAGE_DIR, FASHION200K_ANNOTATION_DIR

class Fashion200K(MyDataset):
	"""
	Fashion200K dataset, introduced in Han et al, Automatic Spatially-aware Fashion Concept Discovery, ICCV'17.
	Each image comes with a compact attribute-like product description (such as black biker jacket or
	wide leg culottes trouser). Queries are created as following: pairs of products that have one word difference
	in their descriptions are selected as the query images and target images; and the modification textis that one
	different word. We used the same training split(around 172k images) and generate queries on the fly for training.

	Class based on TIRG's repo class for the same dataset: https://github.com/google/tirg/
	"""
	
	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):
		MyDataset.__init__(self, split, FASHION200K_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, **kw)

		# get label files for the split
		label_path = FASHION200K_ANNOTATION_DIR
		label_files = [	f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f)) ]
		label_files = [f for f in label_files if split in f]

		# read image info from label files
		self.imgs = []

		def caption_post_process(s):
			return s.strip().replace('.',
									'dotmark').replace('?', 'questionmark').replace(
									'&', 'andmark').replace('*', 'starmark')

		for filename in label_files:
			print('read ' + filename)
			with open(label_path + '/' + filename) as f:
				lines = f.readlines()
			for line in lines:
				line = line.split('	')
				img = {
					'file_path': line[0].replace('women/',''),
					'detection_score': line[1],
					'captions': [caption_post_process(line[2])],
					'split': split,
					'modifiable': False
					}
				self.imgs += [img]
		print('Fashion200K: '+ str(len(self.imgs))+' images')

		# generate query for training or testing
		if split == 'train':
			self.caption_index_init_()
		else:
			self.validate_query_file = os.path.join(FASHION200K_IMAGE_DIR, f'{split}_queries.txt')
			self.generate_test_queries_()

		# generate the list of correct targets for each query
		if what_elements == 'query':
			self.get_all_targets_()


	def get_different_word(self, source_caption, target_caption):
		source_words = source_caption.split()
		target_words = target_caption.split()
		for source_word in source_words:
			if source_word not in target_words:
				break
		for target_word in target_words:
			if target_word not in source_words:
				break
		mod_str = 'replace ' + source_word + ' with ' + target_word
		return source_word, target_word, mod_str


	def generate_test_queries_(self):
		file2imgid = {}
		for i, img in enumerate(self.imgs):
			file2imgid[img['file_path']] = i
		with open(self.validate_query_file) as f:
			lines = f.readlines()
		self.test_queries = []
		for line in lines:
			source_file, target_file = line.split()
			source_file = source_file.replace('women/', '')
			target_file = target_file.replace('women/', '')
			idx = file2imgid[source_file]
			target_idx = file2imgid[target_file]
			source_caption = self.imgs[idx]['captions'][0]
			target_caption = self.imgs[target_idx]['captions'][0]
			source_word, target_word, mod_str = self.get_different_word(
				source_caption, target_caption)
			self.test_queries += [{
				'source_img_id': idx,
				'source_caption': source_caption,
				'target_img_id': target_idx,
				'target_caption': target_caption,
				'mod': {
					'str': mod_str
				}
			}]

	def caption_index_init_(self):
		""" index caption to generate training query-target example on the fly later"""

		# index caption 2 caption_id and caption 2 image_ids
		caption2id = {}
		id2caption = {}
		caption2imgids = {}
		for i, img in enumerate(self.imgs):
			for c in img['captions']:
				if not c in caption2id:
					id2caption[len(caption2id)] = c
					caption2id[c] = len(caption2id)
					caption2imgids[c] = []
				caption2imgids[c].append(i)
		self.caption2imgids = caption2imgids
		print(str(len(caption2imgids))+' unique captions')

		# parent captions are 1-word shorter than their children
		parent2children_captions = {}
		for c in caption2id.keys():
			for w in c.split():
				p = c.replace(w, '')
				p = p.replace('  ', ' ').strip()
				if not p in parent2children_captions:
					parent2children_captions[p] = []
				if c not in parent2children_captions[p]:
					parent2children_captions[p].append(c)
		self.parent2children_captions = parent2children_captions

		# identify parent captions for each image
		for img in self.imgs:
			img['modifiable'] = False
			img['parent_captions'] = []
		for p in parent2children_captions:
			if len(parent2children_captions[p]) >= 2:
				for c in parent2children_captions[p]:
					for imgid in caption2imgids[c]:
						self.imgs[imgid]['modifiable'] = True
						self.imgs[imgid]['parent_captions'] += [p]
		num_modifiable_imgs = 0
		for img in self.imgs:
			if img['modifiable']:
				num_modifiable_imgs += 1
		print('Modifiable images: '+str(num_modifiable_imgs))

	def caption_index_sample_(self, idx):
		while not self.imgs[idx]['modifiable']:
			idx = np.random.randint(0, len(self.imgs))

		# find random target image (same parent)
		img = self.imgs[idx]
		while True:
			p = random.choice(img['parent_captions'])
			c = random.choice(self.parent2children_captions[p])
			if c not in img['captions']:
				break
		target_idx = random.choice(self.caption2imgids[c])

		# find the word difference between query and target (not in parent caption)
		source_caption = self.imgs[idx]['captions'][0]
		target_caption = self.imgs[target_idx]['captions'][0]
		source_word, target_word, mod_str = self.get_different_word(
			source_caption, target_caption)
		return idx, target_idx, source_word, target_word, mod_str

	def get_all_texts(self):
		texts = []
		for img in self.imgs:
			for c in img['captions']:
				texts.append(c)
		return texts

	def get_all_targets_(self):
		all_caps = [img['captions'][0] for img in self.imgs]
		self.all_targets = []
		for q in tqdm(self.test_queries, desc='Creating list of correct targets'):
			target_caption = q['target_caption']
			all_good_imgs  = [i for i, x in enumerate(all_caps) if x == target_caption]
			self.all_targets.append( all_good_imgs )


	def __len__(self):
		if self.what_elements == "query":
			return len(self.test_queries)
		else:
			# - "triplet": len(self.imgs), as in the official code
			# 	of TIRG. This doesn't reflect the actual number of train
			# 	queries but train queries are generated online anyway.
			# - "target": len(self.imgs) is exactly the number of images
			# 	in the gallery
			return len(self.imgs)


	############################################################################
	# *** GET ITEM METHODS
	############################################################################

	def get_triplet(self, index):
		source_idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(index)

		text, real_text = self.get_transformed_captions([mod_str])

		path_src =  self.imgs[source_idx]['file_path']
		img_src = self.get_transformed_image(path_src)

		path_trg =  self.imgs[target_idx]['file_path']
		img_trg = self.get_transformed_image(path_trg)

		return img_src, text, img_trg, real_text, source_idx


	def get_query(self, index):
		test_query = self.test_queries[index]
		source_idx = test_query['source_img_id']
		mod_str = test_query['mod']['str']

		text, real_text = self.get_transformed_captions([mod_str])

		path_src =  self.imgs[source_idx]['file_path']
		img_src = self.get_transformed_image(path_src)

		#target_idx = [test_query['target_img_id']] # TODO: change target_idx to idx of all images w/ the same caption as the target img
		target_idx = self.all_targets[index]

		return img_src, text, source_idx, target_idx, real_text, index


	def get_target(self, index):

		img_id = index
		path_img = self.imgs[img_id]['file_path']
		img = self.get_transformed_image(path_img)

		return img, img_id, index

