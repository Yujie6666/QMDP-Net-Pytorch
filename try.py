#coding:utf-8
from config.config import get_config
import numpy as np
import torch
import os
import Utilsf
import model as model
import importlib
import environment.regular_maze as env


# map_train,b0_train,action_last_train,action_label_train,observation_train,weight_train,map_valid,b0_valid,action_last_valid,action_label_valid,observation_valid,weight_valid = Utilsf.load_dataset('data/regular_maze/train',num_traj=5)
print('model'+str(0))
# a=torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
#
# b=torch.tensor([[2,4,6,8,10],[2,4,6,8,10]])
#
# print(torch.cat((a,b),1))
#print(map_train.shape)