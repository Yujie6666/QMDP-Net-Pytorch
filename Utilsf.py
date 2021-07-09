import sys
import time
import numpy as np
import torch
from config.config import get_config
import random
from tensorboardX import SummaryWriter
import os

def mystr(x):
    x = str(x).split(',')
    res = ''
    for item in x:
        res += item + '\n \n'
    return res

class Tracker:
    def __init__(self, logdir=None):
        if logdir!=None:
            self.writer = SummaryWriter(logdir=logdir)
        else:
            config = get_config()
            if config['run_name']!='default':
                run_name = config['run_name']
            else:
                run_name = config['task_name']+'-'+config['model_name']
            self.writer = SummaryWriter(comment='-' + run_name)
            self.writer.add_text('hyper_parameter', mystr(config))
        self.path = self.writer.logdir
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []

    def train_track(self, reward, frame):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        print('%d: done %d games, reward %.3f, mean reward %.3f, speed %.2f f/s' % (
            frame, len(self.total_rewards), reward, mean_reward, speed))
        sys.stdout.flush()
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)

def test_game(env, agent, tracker, offset, test_games=1, prefix='test'):

    gamma = get_config()['gamma']
    rew_list = []
    succ = 0
    for game in range(test_games):#进行test_games轮游戏
        print("test_game:", game)
        rew_tmp = []
        map,b0,a0,o0 = env.reset()#环境重置
        step = 0
        isstart = True
        action = a0
        observation = o0
        while True:
            step += 1
            action = agent.act(map,b0,action,observation,isstart)#根据当前状态选择动作
            if isstart:
                isstart = False
            observation, reward, done, info = env.step(action)#环境根据动作给出状态、奖励、是否结束、
            rew_tmp.append(reward)#添加当前步骤reward
            if done:
                if reward==1.0:
                    succ += 1#表示成功了
                sum_rew = 0
                for r in reversed(rew_tmp):
                    sum_rew = sum_rew*gamma + r
                #print("game:", game,",sum_rew:",sum_rew)
                tracker.writer.add_scalar("%s_reward"%prefix, sum_rew, offset + game)
                #横坐标第几个游戏，纵坐标游戏的全部折扣奖励
                rew_list.append(sum_rew)#在list中加入本轮的折扣奖励，最后一起求和
                print("game:",game,",sum_reward:",sum_rew,"rew_temp.length:",len(rew_tmp))
                break

    return succ/test_games, np.mean(rew_list), np.std(rew_list)

def load_dataset(path, train_ratio=0.95, step_size=4,num_traj=5):
    ckp = torch.load(path)
    # ckp = torch.load('data/regular_maze2/test')
    Map = ckp['Map']  # 50000个map,但是num_traj个map的障碍部分是一样的，即0~4号map障碍分布相同，但goal不同
    B0 = ckp['B0']  # 50000个初始置信度
    Path = ckp['Path']  # 50000组动作
    goal = ckp['goal']  # 50000组观测
    num = len(Path)
    weight = np.zeros([num, step_size])

    action_last = np.zeros(([num, step_size]))  # 50000*4
    action_label = np.zeros([num, step_size])  # 50000*4
    observation = np.zeros([num, step_size, 4])  # 50000*4*4
    b0 = np.array(B0).reshape(num, 10, 10)
    map_res = np.zeros([num, 10, 10, 2])
    for i in range(num):
        index = int(i / num_traj)
        goal_x = int(goal[i] / 10)
        goal_y = int(goal[i] % 10)
        map_res[i, :, :, 0] = Map[index]
        map_res[i, goal_x, goal_y, 1] = 1
        k = 0
        while (k < len(Path[i]) and k < step_size):
            action_last[i, k] = Path[i][k][1]
            observation[i, k] = np.array(np.unravel_index(Path[i][k][2], [2, 2, 2, 2]), 'i')
            if ((k + 1) < len(Path[i])):
                weight[i, k] = 1
                action_label[i, k] = Path[i][k + 1][1]
            k += 1
    train_num = int(num * train_ratio)  # 选取95%作为训练集，选取5%作为alid集
    map_train = map_res[:train_num]
    b0_train = b0[:train_num]
    action_last_train = action_last[:train_num]
    action_label_train = action_label[:train_num]
    observation_train = observation[:train_num]
    weight_train = weight[:train_num]

    map_valid = map_res[train_num:]
    b0_valid = b0[train_num:]
    action_last_valid = action_last[train_num:]
    action_label_valid = action_label[train_num:]
    observation_valid = observation[train_num:]
    weight_valid = weight[train_num:]

    return map_train,b0_train,action_last_train,action_label_train,observation_train,weight_train,\
           map_valid,b0_valid,action_last_valid,action_label_valid,observation_valid,weight_valid