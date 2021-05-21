from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth



def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Survival_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		n_bins = 4,
		ignore=[],
		label_col = None,
		train_col = None,
		site_col = 'institute',
		multi_site = False,
		filter_dict = {},
		inst = None,
		eps=1e-6,
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.custom_test_ids = None
		self.seed = seed
		self.print_info = print_info
		# self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		
		self.inst = inst


		slide_data = pd.read_csv(csv_path, index_col=0)
		self.institutes = slide_data[site_col].unique()

		if 'case_id' not in slide_data:
			slide_data['case_id'] = slide_data.index
			slide_data = slide_data.reset_index(drop=True)

		if not label_col:
			label_col = 'survival'
		else:
			assert label_col in slide_data.columns
		self.label_col = label_col
		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)


		# all unique patients
		patients_df = slide_data.drop_duplicates(['case_id']).copy()
		uncensored_df = patients_df[patients_df['censorship'] < 1]

		# cut/discretize uncensored patients into quartiles and retrieve bins
		disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
		q_bins[-1] = slide_data[label_col].max() + eps
		q_bins[0] = slide_data[label_col].min() - eps
		
		# cut all patients into bins using the retrieved bin ranges
		disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
		patients_df.insert(2, 'label', disc_labels.values.astype(int))
		self.q_bins = q_bins

		# create dict of case_id: slide_ids
		patient_dict = {}
		slide_data = slide_data.set_index('case_id')
		for patient in patients_df['case_id']:
			slide_ids = slide_data.loc[patient, 'slide_id']
			if isinstance(slide_ids, str):
				slide_ids = np.array(slide_ids).reshape(-1)
			else:
				slide_ids = slide_ids.values
			patient_dict.update({patient:slide_ids})

		self.patient_dict = patient_dict
		
		# doing prediction at patient-level
		slide_data = patients_df
		slide_data.reset_index(drop=True, inplace=True)
		slide_data = slide_data.assign(slide_id=slide_data['case_id'])

		label_dict = {}
		key_count = 0
		for i in range(len(q_bins)-1):
			for c in [0, 1]:
				if multi_site:
					for institute in self.institutes:
						print('(site {}, bin {}, censorship {}): {}'.format(institute, i, c, key_count))
						label_dict.update({(institute, i, c):key_count})
						key_count+=1
				else:
					print('(bin {}, censorship {}): {}'.format(i, c, key_count))
					label_dict.update({(i, c):key_count})
					key_count+=1

		self.label_dict = label_dict
		for i in slide_data.index:
			key = slide_data.loc[i, 'label']
			slide_data.at[i, 'disc_label'] = key
			censorship = slide_data.loc[i, 'censorship']
			if multi_site:
				institute = slide_data.at[i, 'institute']
				key = (institute, key, int(censorship))
			else:
				key = (key, int(censorship))
			slide_data.at[i, 'label'] = label_dict[key]

		self.bins = q_bins
		self.num_classes=len(self.label_dict)
		self.slide_data = slide_data
		

		self.cls_ids_prep()

		if train_col is not None:
			test_ids = np.where(self.slide_data[train_col] == 0)[0]
			self.test_ids = test_ids
		
		if print_info:
			self.summarize()

	def sample_held_out(self, test_num = (50, 50)):

		test_ids = []
		np.random.seed(self.seed) #fix seed
		
		cls_ids = self.slide_cls_ids

		for c in range(len(test_num)):
			test_ids.extend(np.random.choice(cls_ids[c], test_num[c], replace = False)) # validation ids

		return test_ids

	@staticmethod
	def init_multi_site_label_dict(slide_data, label_dict):
		print('initiating multi-source label dictionary')
		sites = np.unique(slide_data['site'].values)
		multi_site_dict = {}
		num_classes = len(label_dict)
		for key, val in label_dict.items():
			for idx, site in enumerate(sites):
				site_key = (key, site)
				site_val = val+idx*num_classes
				multi_site_dict.update({site_key:site_val})
				print('{} : {}'.format(site_key, site_val))
		return multi_site_dict

	def cls_ids_prep(self):
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("patient-level counts: ", self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Class %i:' % i)
			print('Patient-LVL; Number of samples: %d' % (self.slide_cls_ids[i].shape[0]))
			if self.test_ids is not None:
				print('Number of held-out test samples: {}'.format(len(np.intersect1d(self.test_ids, self.slide_cls_ids[i]))))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, split):
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None, no_fl=False):
		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
			
			else:
				test_split = None
			
			return train_split, val_split, test_split
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path)
			val_split = self.get_split_from_df(all_splits['val'])
			test_split = self.get_split_from_df(all_splits['test'])

			train_splits = []
			if no_fl:
				train_split = all_splits['train']
				train_splits.append(self.get_split_from_df(train_split))
			elif self.inst is not None:
				mask = all_splits['train'].isin(self.slide_data[self.slide_data['institute']==self.inst].slide_id)
				train_split = all_splits.loc[mask, 'train']
				train_splits.append(self.get_split_from_df(train_split))
			else:
				for inst in self.institutes:
					mask = all_splits['train'].isin(self.slide_data[self.slide_data['institute']==inst].slide_id)
					train_split = all_splits.loc[mask, 'train']
					train_splits.append(self.get_split_from_df(train_split))
				
			return train_splits, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def key_pair2desc(self, key_pair):
		if len(key_pair) == 2:
			label, censorship = key_pair
			label_desc = (self.bins[label], self.bins[label] + 1, censorship)
		else:
			institute, label, censorship = key_pair
			label_desc = (institute, self.bins[label], self.bins[label] + 1, censorship)
		return label_desc

	def test_split_gen(self, return_descriptor=False):
		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			index = [self.key_pair2desc(key_pair) for key_pair in index]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)
		
		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
		unique = np.append(unique, missing_classes)
		counts = np.append(counts, np.full(len(missing_classes), 0))
		inds = unique.argsort()
		counts = counts[inds]
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
		unique = np.append(unique, missing_classes)
		counts = np.append(counts, np.full(len(missing_classes), 0))
		inds = unique.argsort()
		counts = counts[inds]
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		missing_classes = np.setdiff1d(np.arange(self.num_classes), unique)
		unique = np.append(unique, missing_classes)
		counts = np.append(counts, np.full(len(missing_classes), 0))
		inds = unique.argsort()
		counts = counts[inds]
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		case_id = self.slide_data['case_id'][idx]
		label = self.slide_data['disc_label'][idx]
		event_time = self.slide_data[self.label_col][idx]
		c = self.slide_data['censorship'][idx]

		slide_ids = self.patient_dict[case_id]

		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				all_features = []
				for slide_id in slide_ids:
					full_path = os.path.join(data_dir,'pt_files', '{}.pt'.format(slide_id))
					features = torch.load(full_path)
					all_features.append(features)
				all_features = torch.cat(all_features, dim=0)
				return all_features, label, event_time, c
			else:
				return slide_ids, label, event_time, c

class Generic_Split(Generic_MIL_Survival_Dataset):
	def __init__(self, slide_data, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.label_col=label_col
		self.patient_dict = patient_dict
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


