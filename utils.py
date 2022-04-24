import torch
import torch.nn as nn

def params_require_grad(module, update):
	for idx, param in enumerate(module.parameters()):
		param.requires_grad = update


def l2norm(x):
	"""L2-normalize each row of x"""
	norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
	return torch.div(x, norm)


class SimpleModule(nn.Module):

	def __init__(self, input_dim, embed_dim):
		super(SimpleModule, self).__init__()
		self.linear = nn.Linear(input_dim, embed_dim)
	
	def forward(self, x):
		return self.linear(x)
