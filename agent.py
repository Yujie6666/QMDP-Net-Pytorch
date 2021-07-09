import copy
import numpy as np
import torch
import model
from config.config import get_config

import random
from buffer import Replay
import torch.nn.functional as F

class Actor():

    def __init__(self, main_net, eps=True):
        self.cf = get_config()
        self.main_net = main_net
        if eps:
            self.eps = self.cf['epsilon_max']
            self.eps_delta = (self.cf['epsilon_max']-self.cf['epsilon_min'])/self.cf['epsilon_decrease_step']

    def update_eps(self):
        if self.eps>self.cf['epsilon_min']:
            self.eps -= self.eps_delta
        return self.eps

    def act(self, map, b0, action, observation, isstart):
        map = torch.cuda.FloatTensor(map).view(1,self.cf['imsize'],self.cf['imsize'],2).permute(0,3,1,2)
        b0 = torch.cuda.FloatTensor(b0).view(1,self.cf['imsize'],self.cf['imsize'])
        #print('action:',action)
        action = torch.cuda.LongTensor(action).view(1,1)
        #print('observation:',observation)
        observation = torch.cuda.FloatTensor(observation).view(1,1,self.cf['O_size'])
        with torch.no_grad():
            self.main_net.eval()
            Q,_ = self.main_net(map, b0, action,observation,isstart)
            Q = Q.view(-1)
            #self.main_net.train()
            action = int(Q.max(0)[1].item())
            #print("a_next:",action)
        return np.array([action])

class Learner():

    def __init__(self, ds):
        self.cf = get_config()
        self.main_net = model.UVIN(ds).to('cuda')
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=self.cf['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.buffer = Replay()
        self.loss_vec = []

    def learn(self):

        ma,s1,s2,option,R = self.buffer.sample()
        self.optimizer.zero_grad()
        Q = self.main_net(ma, s1, s2)
        Q = Q.gather(2, option).view(-1)
        loss = torch.nn.MSELoss()(Q, R)
        loss.backward()
        self.optimizer.step()
        loss = float(loss.detach().cpu().numpy())

        self.loss_vec.append(loss)
        if len(self.loss_vec) > self.cf['scheduler_step']:
            self.scheduler.step(np.mean(self.loss_vec))
            self.loss_vec = []

        return loss