import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config.config import get_config
from torch.autograd import Variable

import numpy as np

class POUVIN(nn.Module):
    def __init__(self):
        # 总体模型初始化,传入params相关参数
        super(POUVIN, self).__init__()
        self.cf = get_config()
        self.belief = None
        self.step_size = 1

        #定义转移模型f_T
        self.f_Tb = nn.Conv2d(in_channels=1,out_channels=self.cf['A_size'],kernel_size=3,padding=1,bias=True)
        nn.init.normal_(self.f_Tb.weight, mean=1.0 / 9.0, std=1.0 / 90.0)
        nn.init.constant_(self.f_Tb.bias,0.0)
        self.f_Tv = nn.Conv2d(in_channels=1,out_channels=self.cf['A_size'],kernel_size=3,padding=1,bias=True)
        nn.init.normal_(self.f_Tv.weight, mean=1.0 / 9.0, std=1.0 / 90.0)
        nn.init.constant_(self.f_Tv.bias,0.0)

        #定义贝叶斯模块相关方法
        self.f_Z1 = nn.Conv2d(in_channels=1, out_channels=150, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.f_Z1.weight,mean=0.0,std=1.0 / np.sqrt(9))
        nn.init.constant_(self.f_Z1.bias, 0.0)
        self.f_Z2 = nn.Conv2d(in_channels=150, out_channels=17, kernel_size=1, padding=0, bias=True)
        nn.init.normal_(self.f_Z2.weight,mean=0.0,std=1.0 / np.sqrt(150))
        nn.init.constant_(self.f_Z2.bias, 0.0)
        self.f_Z3 = nn.Sigmoid()

        self.f_O1 = nn.Linear(4, 17)
        nn.init.normal_(self.f_O1.weight,mean=0.0,std=0.25)
        nn.init.constant_(self.f_O1.bias, 0.0)
        self.f_O2 = nn.Tanh()
        self.f_O3 = nn.Linear(17,17)
        nn.init.normal_(self.f_O3.weight, mean=0.0, std=1.0/17)
        nn.init.constant_(self.f_O3.bias, 0.0)
        self.f_O4 = nn.Softmax(dim=1)


        #定义UVIN模块相关方法
        self.fr_1 = torch.nn.Conv2d(in_channels=2, out_channels=150, kernel_size=3, padding=1, bias=True)  # fR函数
        nn.init.normal_(self.fr_1.weight,mean=0.0,std=1.0 / np.sqrt(18))
        nn.init.constant_(self.fr_1.bias, 0.0)

        self.fr_1a = torch.nn.ReLU()

        self.fr_2 = torch.nn.Conv2d(in_channels=150,out_channels=self.cf['A_size'],kernel_size=1)
        nn.init.normal_(self.fr_2.weight,mean=0.0,std=1.0/ np.sqrt(150))
        nn.init.constant_(self.fr_2.bias, 0.0)

        self.final_layer = nn.Linear(self.cf['A_size'], self.cf['A_size'])
        nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.25)
        nn.init.constant_(self.final_layer.bias, 0.0)

    def forward(self, map, b0, act_in, obs_in, is_start,step_size=1):
        self.step_size = step_size
        Q = self.VI(map)
        Z = self.f_Z(map[:,:1,:,:])
        if is_start:
            self.belief = torch.cuda.FloatTensor(b0)  # 此处self.belief设置为Tensor,不需要进行训练
        b = self.belief
        outputs = []
        for step in range(self.step_size):
            b = self.BayesFilter(Z, b, act_in[:,step], obs_in[:,step])
            action_pred = self.policy(Q, b)
            outputs.append(action_pred)
        self.belief = b
        outputs = torch.cat(tuple(outputs), 0)
        return outputs,self.belief

    def f_Z(self,map):
        #从地图中获取，潜在的先验概率Z(o|s),map为batchsize*N*N
        Z1 = self.f_Z1(map)
        Z2 = self.f_Z2(Z1)
        Z3 = self.f_Z3(Z2)
        Z3 = Z3.permute(0,2,3,1)
        Z_sum = torch.sum(Z3, dim=[3], keepdim=True)
        Z = torch.div(Z3, Z_sum + 1e-8)
        return Z #batchsize*imsize*imsize*17

    def BayesFilter(self,Z, b, act_item, obs_item):
        #输入Z是batchsize*N*N*257的，obs_item是batchsize*4的
        now_bs = Z.shape[0]
        w_O1 = self.f_O1(obs_item)
        w_O2 = self.f_O2(w_O1)
        w_O3 = self.f_O3(w_O2)
        w_O4 = self.f_O4(w_O3)
        w_O4 = w_O4.view(now_bs,1,1,17)#batchsize*1*1*17
        Z_O = torch.sum(torch.multiply(Z,w_O4),dim=3,keepdim=False)#Z_O是batch*imsize*imsize的张量

        #将转移模型加入到b中，求得b_prime
        b = b.view(now_bs,1,self.cf['imsize'],self.cf['imsize'])
        b_prime1 = self.f_Tb(b).permute(0,2,3,1)#batchsize*imsize*imsize*A_size

        #选择当前动作对应的b_prime,求得b_prime_a
        #先把动作转为独热编码
        ones = torch.eye(self.cf['A_size'], device='cuda')
        act_item1 = ones.index_select(dim=0, index=act_item)
        act_item2 = act_item1.view(now_bs, 1, 1, self.cf['A_size'])#batchsize*1*1*A_size
        #将独热编码相乘，并进行求和，得到对应位置的结果
        b_prime_a = torch.sum(b_prime1 * act_item2, dim=3)#batch*imsize*imsize
        b_next = b_prime_a*Z_O

        b_next_normal = torch.div(b_next,torch.sum(b_next,dim=[1,2],keepdim=True) + 1e-8)
        return b_next_normal#batch*imsize*imsize


    def VI(self,map):
        R1 = self.fr_1(map)
        R2 = self.fr_1a(R1)
        R = self.fr_2(R2) #batch_size*A_size*imsize*imsize
        v = torch.zeros(map.shape[0],1,self.cf['imsize'],self.cf['imsize'],device='cuda')
        #print(v.shape)
        for k in range(self.cf['VI_k']):
            Q = self.f_Tv(v) #batchsize*A_size*imsize*imsize
            Q = Q + R
            v = torch.max(Q, dim=1,keepdim=True)[0]
        Q = Q.permute(0, 2, 3, 1)
        return Q

    def policy(self,Q,b):
        now_bs = Q.shape[0]
        b = b.view(now_bs,self.cf['imsize'],self.cf['imsize'],1)
        Q_b1 = torch.sum(Q*b, dim=[1,2])
        Q_b = self.final_layer(Q_b1)
        return Q_b

