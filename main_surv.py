from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_surv import train_fl_surv, train_surv
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

from timeit import default_timer as timer

def main(args):
	# create results directory if necessary
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	all_test_auc = []
	all_val_auc = []
	all_test_acc = []
	all_val_acc = []
	all_val_c_index = []
	all_test_c_index = []
	folds = np.arange(start, end)

	for i in folds:
		start = timer()
		seed_torch(args.seed)
		train_datasets, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i), no_fl=args.no_fl)
		
		if len(train_datasets)>1:
			for idx in range(len(train_datasets)):
				print("worker_{} training on {} samples".format(idx,len(train_datasets[idx])))
			print('validation: {}, testing: {}'.format(len(val_dataset), len(test_dataset)))
			datasets = (train_datasets, val_dataset, test_dataset)
			results, test_c_index, val_c_index  = train_fl_surv(datasets, i, args)
		else:
			train_dataset = train_datasets[0] 
			print('training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
			datasets = (train_dataset, val_dataset, test_dataset)
			results, test_c_index, val_c_index  = train_surv(datasets, i, args)

		all_test_c_index.append(test_c_index)
		all_val_c_index.append(val_c_index)

		#write results to pkl
		filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
		save_pkl(filename, results)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))


	final_df = pd.DataFrame({'folds': folds, 'test_c_index': all_test_c_index, 
		'val_c_index': all_val_c_index})

	if len(folds) != args.k:
		save_name = 'summary_partial_{}_{}.csv'.format(start, end)
	else:
		save_name = 'summary.csv'
	final_df.to_csv(os.path.join(args.results_dir, save_name))

# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default='/media/fedshyvana/ssd1', 
					help='data directory')
parser.add_argument('--max_epochs', type=int, default=50,
					help='maximum number of epochs to train')
parser.add_argument('--lr', type=float, default=2e-4,
					help='learning rate (default: 0.0002)')
parser.add_argument('--reg', type=float, default=1e-5,
					help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
					help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--noise_level', type=float, default=0,
                    help='noise level added on the shared weights in federated learning (default: 0)')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
					help='manually specify the set of splits to use, ' 
					+'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--bag_loss', type=str, choices=['ce_surv', 'nll_surv'], default='nll_surv',
					 help='slide-level classification loss function (default: ce)')
parser.add_argument('--alpha_surv', type=float, default=0.15, help='How much to upweight uncensored patients')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--task', type=str)
parser.add_argument('--inst_name', type=str, default=None)
parser.add_argument('--weighted_fl_avg', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--no_fl', action='store_true', default=False, help='train on centralized data')
parser.add_argument('--n_bins', type=int, default=8, help='number of bins to use for event-time discretization')
parser.add_argument('--E', type=int, default=1, help='communication_freq')
args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


### task
print("Experiment Name:", args.exp_code)


def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

args.drop_out=True
args.early_stopping=True
args.model_type='attention_mil'
args.model_size='small'

settings = {'num_splits': args.k, 
			'k_start': args.k_start,
			'k_end': args.k_end,
			'task': args.task,
			'max_epochs': args.max_epochs, 
			'results_dir': args.results_dir, 
			'lr': args.lr,
			'experiment': args.exp_code,
			'seed': args.seed,
			'model_type': args.model_type,
			'model_size': args.model_size,
			"use_drop_out": args.drop_out,
			'n_bins': args.n_bins,
			'E': args.E,
			'opt': args.opt}

if args.inst_name is not None:
	settings.update({'inst_name':args.inst_name})

else:
	settings.update({'noise_level': args.noise_level,
					 'weighted_fl_avg': args.weighted_fl_avg})

args.n_classes = args.n_bins

print('\nLoad Dataset')
if args.task == 'survival':
  dataset = Generic_MIL_Survival_Dataset(csv_path = 'dataset_csv/survival_fl_dummy_dataset.csv',
										   data_dir= os.path.join(args.data_root_dir, 'survival_features_dir'),
										   shuffle = False, 
										   seed = args.seed, 
										   print_info=True,
										   n_bins=args.n_bins,
										   label_col = 'survival_months',
										   inst = args.inst_name,
										   ignore=[])


else:
  raise NotImplementedError

if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)



if args.split_dir is None:
	args.split_dir = os.path.join('./splits', args.task)
else:
	args.split_dir = os.path.join('./splits', args.split_dir)

print("split_dir", args.split_dir)

assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
	print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

if __name__ == "__main__":
	start = timer()
	results = main(args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))


