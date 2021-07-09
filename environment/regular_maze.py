# coding=utf-8

from config.config import get_config
import copy
import numpy as np
import torch
from random import shuffle
from utils import dijkstra


def action2dxy(a):
    return {'0':[0,1],'1':[1,0],'2':[0,-1],'3':[-1,0],'4':[0,0]}.get(str(a),[0,0])

def observation_tolin(obs_bin):
    #因为观测时动作对应的4个方向是否存在障碍物，所以用[0,0,0,0]表示。
    #为了记录方便，在此对其进行转换，从1*8的数组转化为0~255的数字，表示256种不同的观测情况
    return np.ravel_multi_index(obs_bin, np.ones(4, 'i') * 2)

class CoreEnv():

    def __init__(self):
        self.cf = get_config()
        self.reset()

    def step(self, action):
        dxy = action2dxy(action)
        self.current = self.current + dxy

    def reset(self):
        self.current = np.array([0,0], dtype=np.int)
        self.observation = np.zeros(self.cf['O_size'],dtype=np.int)
        #新增部分，在regular中加入了O_size观测空间，可以根据状态获取当前位置的观测

    def set_state(self, state):
        self.current = np.array(state, dtype=np.int)

    def get_state(self):
        return tuple(self.current)

    def get_observation(self,map):
        #根据当前位置和输入的地图获取观测，在当前运动的方向上进行观测
        for action in range(self.cf['O_size']):
            dxy = action2dxy(action)
            S_next = self.current + dxy
            self.observation[action] = self.check_free(map,S_next[0],S_next[1])
        return self.observation

    def check_free(self,map,x,y):
        #判定当前位置是否在地图中，并且当前位置是否有障碍。如果是地图中的空白区域，则返回0，否则返回1
        if (0 <= x and x < self.cf['imsize'] and 0 <= y and y < self.cf['imsize'] and map[x][y] == 0):
            return 0
        return 1



def build_asset():
    #该部分建立图结构<前一状态，当前状态，对应动作>，在此进行保留，沿用学长生成的asset文件
    state_graph = {}
    vis = set()
    env = CoreEnv()
    imsize = get_config()['imsize']
    A_size = get_config()['A_size']

    def dfs(u, fa, faa):
        if not (0<=u[0] and u[0]<imsize and 0<=u[1] and u[1]<imsize):
            return
        if fa!=None:
            if fa not in state_graph.keys():
                state_graph[fa] = {}
            state_graph[fa][u] = faa
        if u in vis:
            return
        else:
            vis.add(u)
            print(u)
        for i in range(A_size):
            env.set_state(u)
            env.step(i)
            v = env.get_state()
            dfs(v, u, i)

    dfs((0,0),None,None)
    edge_count = 0
    for key in state_graph.keys():
        edge_count += len(state_graph[key].keys())
    print('Asset Build')
    torch.save({'state_graph':state_graph,'V_size':len(vis),'E_size':edge_count}, 'environment/asset/'+get_config()['task_name'])

class MapGenerator():

    #需要添加部分：生成多条轨迹traj、初始化b0、返回goal_state、起始状态start_state

    def __init__(self, G):
        self.cf = get_config()
        self.G = G

    def rand_dfs(self, u):
        if u==self.goal:
            return [u]
        if u in self.tmpvis:
            return []
        self.tmpvis.add(u)
        if u not in self.G.keys():
            return []
        vlist = list(self.G[u].keys())
        shuffle(vlist)
        for v in vlist:
            res = self.rand_dfs(v)
            if len(res)!=0:
                return [u]+res
        return []

    def get_rand_path(self):
        self.tmpvis = set()
        return self.rand_dfs(self.start)[:-1]

    def get_map(self,Pobst):
        #该方法调用一个get_rand_path生成一个包含一条路径的地图。其中路径的开始是self.start,路径的终点是self.goal
        #改动部分：因为POMDP问题需要在一个图上生成多个轨迹，所以该方法只返回一个合理的地图，但并不返回start和goal
        #start和goal由方法gen_start_and_goal生成
        self.now_map = np.zeros([self.cf['imsize'], self.cf['imsize']], dtype=np.uint8)

        # borders，格子的四周是障碍
        self.now_map[0, :] = 1
        self.now_map[-1, :] = 1
        self.now_map[:, 0] = 1
        self.now_map[:, -1] = 1

        rand_field = np.random.rand(self.cf['imsize'], self.cf['imsize'])  # 在格子世界中随机生成概率
        self.now_map = np.array(np.logical_or(self.now_map, (rand_field < Pobst)), 'i')  # 如果概率<0.25则设置为障碍，并且原有边界障碍不变

        return self.now_map

    def check_free(self,x,y):
        #判定当前位置是否在地图中，并且当前位置是否有障碍。如果是地图中的空白区域，则返回0，否则返回1
        if (0 <= x and x < self.cf['imsize'] and 0 <= y and y < self.cf['imsize'] and self.now_map[x][y] == 0):
            return 0
        return 1

    def get_observation(self, state):
        #根据当前位置和输入的地图获取观测，在当前运动的方向上进行观测
        observation = np.zeros(self.cf['O_size'],dtype=np.int)
        state = np.array(state)
        for action in range(self.cf['O_size']):
            dxy = action2dxy(action)
            S_next = state + dxy
            observation[action] = self.check_free(S_next[0],S_next[1])
        return observation

    def get_shortest_path(self,start_state,goal_state):
        start_state = tuple(start_state)
        goal_state = tuple(goal_state)
        q = [start_state]
        fa = {}
        faop = {}
        tmpvis = set()
        tmpvis.add(start_state)
        success = False
        while len(q)!=0:
            u = q.pop(0)
            if u == goal_state:
                success = True
                break
            if u not in self.G.keys():
                continue
            valist = sorted(list(self.G[u].items()),key=lambda x:x[1])
            for (v,a) in valist:
                if v not in tmpvis:
                    tmpvis.add(v)
                    if self.now_map[v[0]][v[1]]==0:
                        q.append(v)
                        fa[v] = u
                        faop[v] = a

        states = []
        actions = []
        observations = []
        while(True):
            if u == start_state:
                break
            states.append(list(fa[u]))#s_0,...,s_(goal-1)
            actions.append(faop[u])#a_0,...,a_(goal-1)
            observations.append(self.get_observation(u))
            u = fa[u]
        states.reverse()
        actions.reverse()
        observations.reverse()
        return success,states,actions,observations

    def gen_path(self, min_traj_len=4, maxtrails=1000):
        free_states = np.nonzero((self.now_map == 0).flatten())[0]
        freespace_size = len(free_states)
        b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
        b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
        for trial in range(maxtrails):
            b0size = int(np.random.choice(b0sizes))
            b0ind = np.random.choice(free_states, b0size, replace=False)
            b0_lin = np.zeros([self.cf['imsize']**2])
            b0_lin[b0ind] = 1.0 / b0size
            b0 = b0_lin.reshape(self.cf['imsize'], self.cf['imsize'])

            start_state_lin = np.random.choice(self.cf['imsize']**2, p=b0_lin)
            start_state = np.array(np.unravel_index(start_state_lin, [self.cf['imsize'],self.cf['imsize']]), 'i')
            goal_state_lin = np.random.choice(free_states)
            goal_state = np.array(np.unravel_index(goal_state_lin, [self.cf['imsize'],self.cf['imsize']]), 'i')
            if(start_state_lin == goal_state_lin):
                continue
            #该部分只是简单给出了开始和结束状态，还需要判定路径的存在
            #明天该部分需要继续写       2021-03-26结束标记
            success,states,actions,observations = self.get_shortest_path(start_state,goal_state)

            if success and len(states) >= min_traj_len:
                break
        else:
            #该地图没有符合条件的路径
            raise ValueError
        return b0,start_state,goal_state,states,actions,observations


def build_IL_dataset():

    config = get_config()
    G = torch.load('environment/asset/'+config['task_name'])['state_graph']#G就是图表示
    mg = MapGenerator(G)

    num_map = 10000 #10000
    num_traj = 5 #5 #每个环境生成轨迹的数量
    min_traj_len = 5 #每条轨迹最短的长度，训练BPTT时的step_size一般为4
    imsize = config['imsize']
    X = np.zeros([num_map*num_traj,imsize, imsize,2], dtype=np.uint8)
    B0 = np.zeros([num_map*num_traj,imsize,imsize])
    S = []#大小为num_map*num_traj，用list存储,第num个图第traj个的状态序列是S[num*5+traj],其中0<=num<10000,0<=traj<5
    A = []#同理于S
    O = []#同理于S
    path_data = []#同理于S,但是内部存储结构为[S,A,O]组成的过程序列
    goal_data = []

    for map in range(num_map):
        now_map = mg.get_map()
        print("start generate %d map" %map)
        for traj in range(num_traj):
            #print("start generate %d map %d trajectory"%(map,traj))
            X[map*num_traj+traj,:,:,0] = now_map
            b0,start_state,goal_state,states,actions,observations = mg.gen_path(min_traj_len=min_traj_len)
            #print("b0:",b0,"start_state:",start_state,"goal_state:",goal_state,"states",states,"actions",actions,"observations",observations)
            X[map*num_traj+traj,goal_state[0],goal_state[1],1] = 10
            B0[map*num_traj+traj] = b0
            S.append(states)
            A.append(actions)
            O.append(observations)
            goal_data.append(goal_state)
            path_states = np.array(states)
            path_states = path_states[:,0]*imsize+path_states[:,1]
            path_actions = np.array(actions)
            path_observations = np.array(observations)
            path = np.stack([path_states,path_actions,path_observations], axis=1)
            #print(path)
            path_data.append(path)
            #print(X[map,traj,:,:,0])
        #print(S,A,O)
    torch.save({'X':X,'B0':B0,'S':S,'A':A,'O':O,'goal':goal_data,'path_data':path_data}, 'IL_dataset/'+config['task_name'])






class Env():

    def __init__(self):
        self.cf = get_config()
        self.mg = MapGenerator(G=torch.load('environment/asset/' + self.cf['task_name'])['state_graph'])
        self.core = CoreEnv()

    def reset(self):
        map = self.mg.get_map(Pobst=0.25)
        self.X = np.zeros([self.cf['imsize'], self.cf['imsize'],2], dtype=np.uint8)
        b0, start_state, goal_state, _, actions, observations = self.mg.gen_path()
        self.X[:, :, 0] = map
        self.X[goal_state[0], goal_state[1], 1] = 1
        self.left_step = 250
        self.core.set_state(start_state)
        self.goal = tuple(goal_state)
        #print('goal:',self.goal)
        a0 = 4
        o0 = self.core.get_observation(self.X[:,:,0])
        return self.X, b0, np.array([a0]), np.array([o0])
        
    def step(self, action):
        print('action:',action)
        print('state:',self.core.get_state())
        self.core.step(action[0])
        xy = self.core.get_state()
        observation = self.core.get_observation(self.X[:,:,0])

        if not (0<=xy[0] and xy[0]<self.cf['imsize'] and 0<=xy[1] and xy[1]<self.cf['imsize']):
            return observation, -1.0, True, None

        print('xy:',xy,'goal:',self.goal)
        if xy==self.goal:
            r, d = 1.0, True
        elif self.X[xy[0],xy[1],0]==1:
            r, d = -1.0, True
        else:
            r, d = 0.0, False

        self.left_step -= 1
        if self.left_step<=0:
            d = True
        
        return np.array([observation]), r, d, None