#coding:utf-8
from config.config import get_config
import numpy as np
import torch
import os
import Utilsf
import model
import importlib
from agent import Actor
import environment.regular_maze as env

config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_device']
environment = importlib.import_module('environment.'+config['task_name'])
map_test,b0_test,action_last_test,action_label_test,observation_test,weight_test,_,_,_,_,_,_ = Utilsf.load_dataset('data/'+config['task_name']+'/test',train_ratio=1 ,num_traj=1)
net = model.POUVIN().to('cuda')

Loss = torch.nn.CrossEntropyLoss(reduction='none')
ckp = torch.load('runs/0regular/model181')
# ckp = torch.load('runs/1chess/model')
net.load_state_dict(ckp['net'])
# optimizer.load_state_dict(ckp['optimizer'])
step_size = config['step_size']
tracker = Utilsf.Tracker()
ac = 0
psum = 0
for i in range(0, map_test.shape[0], config['batch_size']):
    j = i + config['batch_size']
    if j > map_test.shape[0]: break
    x = torch.cuda.FloatTensor(map_test[i:j]).permute(0, 3, 1, 2)
    b0 = torch.cuda.FloatTensor(b0_test[i:j])
    a_in = torch.cuda.LongTensor(action_last_test[i:j])
    o_in = torch.cuda.FloatTensor(observation_test[i:j])
    a_label = action_label_test[i:j]
    Q, belief = net(x, b0, a_in, o_in, True, step_size)
    belief.detach_()
    label = a_label.T.flatten()
    pre = Q.view(-1,config['A_size']).max(1)[1].cpu().numpy()
    #print('pre\n',pre)
    #print('label\n',label)
    ac += np.where(pre==label)[0].shape[0]
    #print(ac)
    psum += label.shape[0]
print('PA:', ac/psum)#ac是指预测方向和label相同的数量



agent = Actor(net, eps=False)#获取智能体
env = environment.Env()#获取环境
sr,mean,std = Utilsf.test_game(env, agent, tracker, 0, config['test_final_game'])#sr成功率，mean奖励的均值，std奖励的方差
print('test sr:%f, reward: mean:%f std:%f'%(sr, mean, std))#sr指的是最终达到终点的频数，





