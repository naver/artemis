
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossModule(nn.Module):

	def __init__(self, opt):
		super(LossModule, self).__init__()

	def forward(self, scores):
		"""
		Loss based on Equation 6 from the TIRG paper,
		"Composing Text and Image for Image Retrieval - An Empirical Odyssey", CVPR19
		Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
		
		Args:
			scores: matrix of size (batch_size, batch_size), where coefficient
				(i,j) corresponds to the score between query (i) and target (j).
				Ground truth associations are represented along the diagonal.
		"""

		# build the ground truth label tensor: the diagonal corresponds to
		# correct classification
		GT_labels = torch.arange(scores.shape[0]).long()
		GT_labels = torch.autograd.Variable(GT_labels)
		if torch.cuda.is_available():
			GT_labels = GT_labels.cuda()

		# compute the cross-entropy loss
		loss = F.cross_entropy(scores, GT_labels, reduction = 'mean')

		return loss