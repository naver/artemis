import torch
import torch.nn as nn

from model import BaseModel
from utils import l2norm

class ConCatModule(nn.Module):

	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x

class TIRG(BaseModel):
	"""
	The TIRG model.

	Implementation derived (except for BaseModel-inherence) from
	https://github.com/google/tirg (downloaded on July 23th 2020).

	The method is described in Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia
	Li, Li Fei-Fei, James Hays. "Composing Text and Image for Image Retrieval -
	An Empirical Odyssey" CVPR 2019. arXiv:1812.07119
	"""

	def __init__(self, word2idx, opt):
		super(TIRG, self).__init__(word2idx, opt)

		# --- modules
		self.a = nn.Parameter(torch.tensor([1.0, 1.0])) # changed the second coeff from 10.0 to 1.0
		self.gated_feature_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(2 * self.embed_dim), nn.ReLU(),
				nn.Linear(2 * self.embed_dim, self.embed_dim))
		self.res_info_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(2 * self.embed_dim), nn.ReLU(),
				nn.Linear(2 * self.embed_dim, 2 * self.embed_dim), nn.ReLU(),
				nn.Linear(2 * self.embed_dim, self.embed_dim))


	def query_compositional_embedding(self, main_features, modifying_features):
		f1 = self.gated_feature_composer((main_features, modifying_features))
		f2 = self.res_info_composer((main_features, modifying_features))
		f = torch.sigmoid(f1) * main_features * self.a[0] + f2 * self.a[1]
		f = l2norm(f) # added to the official TIRG code
		return f


	############################################################################
	# *** SCORING METHODS
	############################################################################

	# 2 versions of scoring methods:
	# - a "regular" version, which returns a tensor of shape (batch_size), where
	#   coefficient (i) is the score between query (i) and target (i). 
	# - a broadcast version, which returns a tensor of shape (batch_size,
	#   batch_size), corresponding to the score matrix where coefficient (i,j)
	#   is the score between query (i) and target (j).

	def compute_score_broadcast(self, r, m, t):
		return self.query_compositional_embedding(r, m).mm(t.t())

	def compute_score(self, r, m, t):
		return (self.query_compositional_embedding(r, m) * t).sum(-1)