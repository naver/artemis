#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import shutil
import time
import pickle
import torch

from option import parser, verify_input_args
import data
from vocab import Vocabulary # necessary import
from artemis_model import ARTEMIS
from tirg_model import TIRG
from loss import LossModule
from evaluate import validate
from logger import AverageMeter
import logging


################################################################################
# *** UTILS
################################################################################

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
										datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def resume_from_ckpt_saved_states(args, model, optimizer):
	"""
	Load model, optimizer, and previous best score.
	"""

	# load checkpoint
	assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
	ckpt = torch.load(args.ckpt)
	print(f"Loading file {args.ckpt}.")

	# load model
	if torch.cuda.is_available():
		model.load_state_dict(ckpt['model'])
	else :
		state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
		model.load_state_dict(state_dict)
	print("Model: resume from provided state.")

	# load the optimizer state
	optimizer.load_state_dict(ckpt['optimizer'])
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v
				if torch.cuda.is_available():
					state[k] = state[k].cuda()
	print("Optimizer: resume from provided state.")

	# load the previous best score
	best_score = ckpt['best_score']
	print("Best score: obtained from provided state.")

	return model, optimizer, best_score


################################################################################
# *** TRAINING FOR ONE EPOCH
################################################################################

def train_model(epoch, data_loader, model, criterion, optimizer, args):

	# Switch to train mode
	model.train()

	# Average meter to record the training statistics
	loss_info = AverageMeter(precision=8) # precision: number of digits after the comma

	max_itr = len(data_loader)
	for itr, data in enumerate(data_loader):

		# Get data
		img_src, txt, txt_len, img_trg, _, _ = data
		if torch.cuda.is_available():
			img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()

		# Forward pass
		scores = model.forward_broadcast(img_src, img_trg, txt, txt_len)
		# rescale the scores for training optimization purpose
		if args.learn_temperature:
			scores *= model.temperature.exp()

		# Compute loss
		loss = criterion(scores)
		# update the loss statistics
		loss_info.update(loss.item())

		# Backprop
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Print log info
		if itr > 0 and (itr % args.log_step == 0 or itr + 1 == max_itr):
			log_msg = 'loss: %s' % str(loss_info)
			logging.info('[%d][%d/%d] %s' %(epoch, itr, max_itr, log_msg))

	return loss_info.avg


################################################################################
# *** VALIDATE
################################################################################

def validate_model(model, args, vocab, epoch=-1, best_score=None, split='val'):

	# Switch to eval mode
	model.eval()

	with torch.no_grad():
		start = time.time()
		message, val_mes = validate(model, args, vocab, split=split)
		end = time.time()

	log_msg = "[%s][%d] >> EVALUATION <<" % (args.exp_name, epoch)
	log_msg += "\nProcessing time : %f" % (end - start)
	log_msg += message

	if best_score:
		log_msg += '\nCurrent best score: %.2f' %(best_score)

	logging.info(log_msg)

	return val_mes

def update_best_score(new_score, old_score, is_higher_better=True):
	if not old_score:
		score, updated = new_score, True
	else:
		if is_higher_better:
			score = max(new_score, old_score)
			updated = new_score > old_score
		else:
			score = min(new_score, old_score)
			updated = new_score < old_score
	return score, updated

def save_ckpt(state, is_best, args, filename='ckpt.pth', split='val'):
	ckpt_path = os.path.join(args.ckpt_dir, args.exp_name, filename)
	torch.save(state, ckpt_path)
	if is_best:
		model_best_path =  os.path.join(args.ckpt_dir, args.exp_name, split, 'model_best.pth')
		shutil.copyfile(ckpt_path, model_best_path)
		logging.info('Updating the best model checkpoint: {}'.format(model_best_path))


################################################################################
# *** MAIN
################################################################################

def main():

	# Parse & correct arguments
	args = verify_input_args(parser.parse_args())
	print(args)

	# Load vocabulary
	vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
	assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
	vocab = pickle.load(open(vocab_path, 'rb'))

	# Setup model
	if args.model_version == "TIRG":
		model = TIRG(vocab.word2idx, args)
	else:
		# model version is ARTEMIS or one of its ablatives
		model = ARTEMIS(vocab.word2idx, args)
	print("Model version:", args.model_version)

	# Load the model on GPU
	if torch.cuda.is_available():
		model = model.cuda()
		torch.backends.cudnn.benchmark = True

	# Instanciate the optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	
	# Optionally resume from provided checkpoint
	best_score = {split:None for split in args.validate}
	if args.ckpt:
		model, optimizer, best_score = resume_from_ckpt_saved_states(args, model, optimizer)
		# evaluate after resuming
		for split in args.validate:
			print("\nValidating on the {} split.".format(split))
			with torch.no_grad():
				_ = validate_model(model, args, vocab, -1, best_score[split], split=split) 

	# Instanciate the loss
	criterion = LossModule(args)

	# Dataloaders
	trn_loader = data.get_train_loader(args, vocab)

	# Eventually, train the model!
	for epoch in range(args.num_epochs):

		# decay learning rate epoch
		if epoch != 0 and epoch % args.step_lr == 0:
			for g in optimizer.param_groups:
				print("Learning rate: {} --> {}\n".format(g['lr'], g['lr']*args.gamma_lr))
				g['lr'] *= args.gamma_lr

		# train for one epoch
		train_model(epoch, trn_loader, model, criterion, optimizer, args)

		# evaluate the model & save state if best
		for split in args.validate:
			print("Validating on the {} split.".format(split))

			# evaluate the current split
			with torch.no_grad():
				val_score = validate_model(model, args, vocab, epoch, best_score[split], split=split)

			# remember best validation score
			best_score[split], updated = update_best_score(val_score, best_score[split])

			# save ckpt
			save_ckpt({
				'args': args,
				'epoch': epoch,
				'best_score': best_score,
				'model': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			}, updated, args, split=split)

		print("")

if __name__ == '__main__':
	main()