#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import torch
import torch.nn as nn
import torchtext
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence

from utils import l2norm
from config import TORCH_HOME, GLOVE_DIR

os.environ['TORCH_HOME'] = TORCH_HOME


def get_cnn(arch):
	return torchvision.models.__dict__[arch](pretrained=True)


class GeneralizedMeanPooling(nn.Module):
	"""
	Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.

	The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

		- At p = infinity, one gets Max Pooling
		- At p = 1, one gets Average Pooling

	The output is of size H x W, for any input size.
	The number of output features is equal to the number of input planes.

	Args:
		output_size: the target output size of the image of the form H x W.
					 Can be a tuple (H, W) or a single H for a square image H x H
					 H and W can be either a ``int``, or ``None`` which means the size will
					 be the same as that of the input.
	"""

	def __init__(self, norm, output_size=1, eps=1e-6):
		super(GeneralizedMeanPooling, self).__init__()
		assert norm > 0
		self.p = float(norm)
		self.output_size = output_size
		self.eps = eps

	def forward(self, x):
		x = x.clamp(min=self.eps).pow(self.p)
		return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			+ str(self.p) + ', ' \
			+ 'output_size=' + str(self.output_size) + ')'


class EncoderImage(nn.Module):

	def __init__(self, opt):
		super(EncoderImage, self).__init__()

		# general parameters
		embed_dim = opt.embed_dim
		self.gradcam = opt.gradcam

		# backbone CNN
		self.cnn = get_cnn(opt.cnn_type)
		self.cnn_dim = self.cnn.fc.in_features
		self.pool_dim = self.cnn_dim # for avgpool resnet, cnn output and pooling output have the same dim

		# replace the avgpool and the last fc layer of the CNN with identity
		# then stack new pooling and fc layers at the end
		self.cnn.avgpool = nn.Sequential()
		self.cnn.fc = nn.Sequential()
		self.gemp = GeneralizedMeanPooling(norm=3)
		self.fc = nn.Linear(self.pool_dim, embed_dim)

		# initialize placeholders for heatmap processing
		if opt.gradcam :
			# placeholder for activation maps
			self.activations = None
			# placeholder for the gradients
			self.gradients = None

	@property
	def dtype(self):
		return self.cnn.conv1.weight.dtype

	def forward(self, images):
		# backbone forward
		out_7x7 = self.cnn( images ).view(-1, self.cnn_dim, 7, 7).type(self.dtype)

		# save intermediary results for further studies (gradcam)
		if self.gradcam:
			out_7x7.requires_grad_(True)
			# register activations
			self.register_activations(out_7x7)
			# register gradients
			h = out_7x7.register_hook(self.activations_hook)

		# encoder ending's forward
		out = self.gemp(out_7x7).view(-1, self.pool_dim) # pooling
		out = self.fc(out)
		out = l2norm(out)
		return out

	# --- gradcam utilitary methods for heatmap processing ---

	def activations_hook(self, grad):
		""" hook for the gradients of the activations """
		self.gradients = grad

	def register_activations(self, activations):
		self.activations = activations

	def get_gradient(self):
		""" gradient extraction """
		return self.gradients

	def get_activation(self):
		""" activation extraction """
		return self.activations


class EncoderText(nn.Module):

	def __init__(self, word2idx, opt):
		super(EncoderText, self).__init__()

		wemb_type, word_dim, embed_dim = \
			opt.wemb_type, opt.word_dim, opt.embed_dim

		self.txt_enc_type = opt.txt_enc_type
		self.embed_dim = embed_dim

		# Word embedding
		self.embed = nn.Embedding(len(word2idx), word_dim)

		# Sentence embedding
		if self.txt_enc_type == "bigru":
			self.sent_enc = nn.GRU(word_dim, embed_dim//2, bidirectional=True, batch_first=True)
			self.forward = self.forward_bigru
		elif self.txt_enc_type == "lstm":
			self.lstm_hidden_dim = opt.lstm_hidden_dim
			self.sent_enc = nn.Sequential(
				nn.LSTM(word_dim, self.lstm_hidden_dim),
				nn.Dropout(p=0.1),
				nn.Linear(self.lstm_hidden_dim, embed_dim),
			)
			self.forward =  self.forward_lstm

		self.init_weights(wemb_type, word2idx, word_dim)


	def init_weights(self, wemb_type, word2idx, word_dim):
		if wemb_type is None:
			print("Word embeddings randomly initialized with xavier")
			nn.init.xavier_uniform_(self.embed.weight)
		else:
			# Load pretrained word embedding
			if 'glove' == wemb_type.lower():
				wemb = torchtext.vocab.GloVe(cache=GLOVE_DIR)
			else:
				raise Exception('Unknown word embedding type: {}'.format(wemb_type))
			assert wemb.vectors.shape[1] == word_dim

			# Get word embeddings + keep track of missing words
			missing_words = []
			for word, idx in word2idx.items():
				if word in wemb.stoi:
					self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
				else:
					missing_words.append(word)
			print('Words: {}/{} found in vocabulary; {} words missing'.format(
				len(word2idx)-len(missing_words), len(word2idx), len(missing_words)))

	@property
	def dtype(self):
		return self.embed.weight.data.dtype

	@property
	def device(self):
		return self.embed.weight.data.device

	def forward_bigru(self, x, lengths):

		# embed word ids to vectors
		wemb_out = self.embed(x)

		# for pytorch >= 1.7, length.device == 'cpu' (but it worked as a gpu variable in 1.2)
		lengths = lengths.cpu()

		# forward propagate RNNs
		packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
		if torch.cuda.device_count() > 1:
			self.sent_enc.flatten_parameters()

		_, rnn_out = self.sent_enc(packed)
		# reshape output to (batch_size, hidden_size)
		rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_dim)

		out = l2norm(rnn_out)
		return out

	def forward_lstm(self, x, lengths):

		# embed word ids to vectors
		wemb_out = self.embed(x) # size (batch, max_length, word_dim)
		wemb_out = wemb_out.permute(1, 0, 2) # size (max_length, batch, word_dim)

		# lstm
		batch_size = wemb_out.size(1)
		first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
						torch.zeros(1, batch_size, self.lstm_hidden_dim))
		if torch.cuda.is_available():
			first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
		lstm_output, last_hidden = self.sent_enc[0](wemb_out, first_hidden)

		# extract features
		text_features = []
		for i in range(batch_size):
			text_features.append(lstm_output[:, i, :].max(0)[0])
		text_features = torch.stack(text_features)

		# output
		out = self.sent_enc[1:](text_features)
		out = l2norm(out)
		return out