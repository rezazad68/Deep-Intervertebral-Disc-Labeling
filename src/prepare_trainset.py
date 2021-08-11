
# Author: Lucas
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Revised by Reza Azad

from Data2array import *
from train_utils import *
from models import *
import numpy as np
import copy
import pickle
import argparse

parser = argparse.ArgumentParser(description='PyTorch arguments')
parser.add_argument('--source', default='../dataset/', type=str,
                        help='Pth to Dataset')                     
parser.add_argument('--target', default='./prepared_dataset', type=str,
                        help='Pth to save the prepared dataset')  
parser.add_argument('--mode', default=2, type=int,
                        help='#0 for both t1 and t2 , 1 for t1 only , 2 for t2 only')
                        
args = parser.parse_args()                        
                        
print('load dataset')
ds = load_Data_Bids2Array(args.source, mode= args.mode, split='train', aim='full')
print('creating heatmap')
full = extract_groundtruth_heatmap(ds)
print('saving the prepared dataset')
with open(args.target, 'wb') as file_pi:
     pickle.dump(full, file_pi)
print('finished')   

