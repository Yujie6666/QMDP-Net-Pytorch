#coding:utf-8
from config.config import get_config
import numpy as np
import torch
import os
import Utilsf
import model
import importlib
import environment.regular_maze as env

config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config['visible_device']
environment = importlib.import_module('environment.'+config['task_name'])
map_train,b0_train,action_last_train,action_label_train,observation_train,weight_train,map_valid,b0_valid,action_last_valid,action_label_valid,observation_valid,weight_valid = Utilsf.load_dataset('data/'+config['task_name']+'/train',num_traj=5)

net = model.POUVIN().to('cuda')

Loss = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.RMSprop(net.parameters(), lr=config['learning_rate'])



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.9)
tracker = Utilsf.Tracker()
step_size = config['step_size']
min_trainloss = 0.7
min_validloss = 0.7
for epoch in range(config['max_training_epoch']):
    print('epoch:',epoch,'lr:',optimizer.state_dict()['param_groups'][0]['lr'])
    train_loss = 0.0
    total = map_train.shape[0]/config['batch_size']
    for i in range(0, map_train.shape[0], config['batch_size']):
        j = i + config['batch_size']
        #print("i=",i,"j=",j,"X_train.shape[0]",X_train.shape[0])
        if j > map_train.shape[0]: break
        optimizer.zero_grad()  # 梯度清零，把上一轮的梯度清零，避免叠加
        x = torch.cuda.FloatTensor(map_train[i:j]).permute(0, 3, 1, 2)
        b0 = torch.cuda.FloatTensor(b0_train[i:j])
        a_in = torch.cuda.LongTensor(action_last_train[i:j])
        o_in = torch.cuda.FloatTensor(observation_train[i:j])
        a_label = torch.cuda.LongTensor(action_label_train[i:j])
        Q,belief = net(x,b0,a_in,o_in,True,step_size)
        belief.detach_()
        label = a_label.T.flatten()
        weight = torch.cuda.LongTensor(weight_train[i:j])
        weight = weight.T.flatten()#按照time连接标签
        loss = Loss(Q, label)
        loss = loss*weight
        loss = torch.mean(loss)
        loss.backward()  # loss传递
        #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)#后加的梯度裁剪
        optimizer.step()  # 更新参数
        train_loss += loss / total
        #print('loss',loss)
    if epoch!=0 and epoch%200==0:#each 50 epoch,print grad
        for name, parms in net.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                  ' -->grad_value:', parms.grad)
    scheduler.step()
    valid_loss = 0.0
    total_v = map_valid.shape[0] / config['batch_size']
    for i in range(0, map_valid.shape[0], config['batch_size']):
        j = i + config['batch_size']
        #print("i=",i,"j=",j,"X_train.shape[0]",X_train.shape[0])
        if j > map_valid.shape[0]: break
        x_v = torch.cuda.FloatTensor(map_valid[i:j]).permute(0, 3, 1, 2)
        b0_v = torch.cuda.FloatTensor(b0_valid[i:j])
        a_v = torch.cuda.LongTensor(action_last_valid[i:j])
        o_v = torch.cuda.FloatTensor(observation_valid[i:j])
        av_label = torch.cuda.LongTensor(action_label_valid[i:j])
        Q_v, belief_v = net(x_v, b0_v, a_v, o_v, True, step_size)
        label_v = av_label.T.flatten()
        weight_v = torch.cuda.LongTensor(weight_valid[i:j])
        weight_v = weight_v.T.flatten()  # 按照time连接标签
        loss_v = Loss(Q_v,label_v)
        loss_v = loss_v * weight_v
        loss_v = torch.mean(loss_v)
        valid_loss += loss_v / total_v
    print('train_loss:', train_loss, ',valid_loss:', valid_loss)  # ac是指预测方向和label相同的数量
    tracker.writer.add_scalars("loss", {'train_loss': float(train_loss), 'valid_loss': valid_loss}, epoch)
    if float(train_loss)< min_trainloss and float(valid_loss)< min_validloss:
        model_path = os.path.join(tracker.path, 'model'+str(epoch))
        torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_path)
        min_trainloss = float(train_loss)
        min_validloss = float(valid_loss)
model_path = os.path.join(tracker.path, 'model')
torch.save({'net':net.state_dict(),'optimizer':optimizer.state_dict()}, model_path)


# agent = Actor(net, eps=False)#获取智能体
# env = environment.Env()#获取环境
# sr,mean,std = utils.test_game(env, agent, tracker, 0, config['test_final_game'])#sr成功率，mean奖励的均值，std奖励的方差
# print('test sr:%f, reward: mean:%f std:%f'%(sr, mean, std))#sr指的是最终达到终点的频数，





