#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import torch
import torch.nn as nn

from model import BaseModel
from utils import l2norm

class L2Module(nn.Module):

	def __init__(self):
		super(L2Module, self).__init__()

	def forward(self, x):
		x = l2norm(x)
		return x

class AttentionMechanism(nn.Module):
	"""
	Module defining the architecture of the attention mechanisms in ARTEMIS.
	"""

	def __init__(self, opt):
		super(AttentionMechanism, self).__init__()

		self.embed_dim = opt.embed_dim
		input_dim = self.embed_dim

		self.attention = nn.Sequential(
			nn.Linear(input_dim, self.embed_dim),
			nn.ReLU(),
			nn.Linear(self.embed_dim, self.embed_dim),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		return self.attention(x)

class ARTEMIS(BaseModel):
	"""
	ARTEMIS: Attention-based Retrieval with Text-Explicit Matching and Implicit Similarity,
	ICLR 2022
	"""

	def __init__(self, word2idx, opt):
		super(ARTEMIS, self).__init__(word2idx, opt)

		# --- modules
		self.Transform_m = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), L2Module())
		self.Attention_EM = AttentionMechanism(opt)
		self.Attention_IS = AttentionMechanism(opt)

		# --- scoring strategy
		self.model_version = opt.model_version
		if self.model_version == "ARTEMIS":
			self.compute_score = self.compute_score_artemis
			self.compute_score_broadcast = self.compute_score_broadcast_artemis
		elif self.model_version == "EM-only":
			self.compute_score = self.compute_score_EM
			self.compute_score_broadcast = self.compute_score_broadcast_EM
		elif self.model_version == "IS-only":
			self.compute_score = self.compute_score_IS
			self.compute_score_broadcast = self.compute_score_broadcast_IS
		elif self.model_version == "late-fusion":
			self.compute_score = self.compute_score_arithmetic
			self.compute_score_broadcast = self.compute_score_broadcast_arithmetic
		elif self.model_version == "cross-modal":
			self.compute_score = self.compute_score_crossmodal
			self.compute_score_broadcast = self.compute_score_broadcast_crossmodal
		elif self.model_version == "visual-search":
			self.compute_score = self.compute_score_visualsearch
			self.compute_score_broadcast = self.compute_score_broadcast_visualsearch

		# --- for heatmap processing
		self.gradcam = opt.gradcam
		self.hold_results = dict() # holding intermediate results


	############################################################################
	# *** SCORING METHODS
	############################################################################

	# All scoring methods exist in 2 versions:
	# - a "regular" version, which returns a tensor of shape (batch_size), where
	#   coefficient (i) is the score between query (i) and target (i). 
	# - a broadcast version, which returns a tensor of shape (batch_size,
	#   batch_size), corresponding to the score matrix where coefficient (i,j)
	#   is the score between query (i) and target (j).

	# Shape notations in comments:
	# 	Bq: "query" batch size (number of queries) [in practice, Bq = B]
	# 	Bt: "target" batch size (number of targets) [in practice, Bt = B]
	# 	d: embedding dimension

	def apply_attention(self, a, x):
		return l2norm(a * x)
	
	def compute_score_artemis(self, r, m, t, store_intermediary=False):
		EM = self.compute_score_EM(r, m, t, store_intermediary)
		IS = self.compute_score_IS(r, m, t, store_intermediary)
		if store_intermediary:
			self.hold_results["EM"] = EM
			self.hold_results["IS"] = IS
		return EM + IS
	def compute_score_broadcast_artemis(self, r, m, t):
		return self.compute_score_broadcast_EM(r, m, t) + self.compute_score_broadcast_IS(r, m, t)

	def compute_score_EM(self, r, m, t, store_intermediary=False):
		Tr_m = self.Transform_m(m)
		A_EM_t = self.apply_attention(self.Attention_EM(m), t)
		if store_intermediary:
			self.hold_results["Tr_m"] = Tr_m
			self.hold_results["A_EM_t"] = A_EM_t
		return (Tr_m * A_EM_t).sum(-1)
	def compute_score_broadcast_EM(self, r, m, t):
		batch_size = r.size(0)
		A_EM = self.Attention_EM(m) # shape (Bq, d)
		Tr_m = self.Transform_m(m) # shape (Bq, d)
		# apply each query attention mechanism to all targets
		A_EM_all_t = self.apply_attention(A_EM.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim)) # shape (Bq, Bt, d)
		EM_score = (Tr_m.view(batch_size, 1, self.embed_dim) * A_EM_all_t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j
		return EM_score

	def compute_score_IS(self, r, m, t, store_intermediary=False):
		A_IS_r = self.apply_attention(self.Attention_IS(m), r)
		A_IS_t = self.apply_attention(self.Attention_IS(m), t)
		if store_intermediary:
			self.hold_results["A_IS_r"] = A_IS_r
			self.hold_results["A_IS_t"] = A_IS_t
		return (A_IS_r * A_IS_t).sum(-1)
	def compute_score_broadcast_IS(self, r, m, t):
		batch_size = r.size(0)
		A_IS = self.Attention_IS(m) # shape (Bq, d)
		A_IS_r = self.apply_attention(A_IS, r) # shape (Bq, d)
		# apply each query attention mechanism to all targets
		A_IS_all_t = self.apply_attention(A_IS.view(batch_size, 1, self.embed_dim), t.view(1, batch_size, self.embed_dim)) # shape (Bq, Bt, d)
		IS_score = (A_IS_r.view(batch_size, 1, self.embed_dim) * A_IS_all_t).sum(-1) # shape (Bq, Bt) ; coefficient (i,j) is the IS score between query i and target j
		return IS_score

	def compute_score_arithmetic(self, r, m, t, store_intermediary=False):
		return (l2norm(r + m) * t).sum(-1)
	def compute_score_broadcast_arithmetic(self, r, m, t):
		return (l2norm(r + m)).mm(t.t())

	def compute_score_crossmodal(self, r, m, t, store_intermediary=False):
		return (m * t).sum(-1)
	def compute_score_broadcast_crossmodal(self, r, m, t):
		return m.mm(t.t())

	def compute_score_visualsearch(self, r, m, t, store_intermediary=False):
		return (r * t).sum(-1)
	def compute_score_broadcast_visualsearch(self, r, m, t):
		return r.mm(t.t())


	############################################################################
	# *** FOR HEATMAP PROCESSING
	############################################################################

	def forward_save_intermediary(self, images_src, images_trg, sentences, lengths):

		# clean previously stored results, if any
		self.hold_results.clear()

		# compute embeddings & store activation map if in gradcam mode for heatmap visualizations
		r = self.get_image_embedding(images_src)
		if self.gradcam:
			self.hold_results["r_activation"] = self.img_enc.get_activation()
		t = self.get_image_embedding(images_trg)
		if self.gradcam:
			self.hold_results["t_activation"] = self.img_enc.get_activation()
		m = self.get_txt_embedding(sentences, lengths)

		return self.compute_score(r, m, t, store_intermediary=True)