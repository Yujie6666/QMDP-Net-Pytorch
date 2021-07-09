# coding=utf-8
import argparse, os, random, shutil
import numpy as np, scipy.sparse
import torch

from utils import dijkstra
from utils.dotdict import dotdict
from utils.qmdp import QMDP

try:
    import ipdb as pdb
except Exception:
    import pdb


FREESTATE = 0.0
OBSTACLE = 1.0


class GridBase(object):
    #格子世界基础
    def __init__(self, params):
        """
        Initialize domain simulator
        :param params: domain descriptor dotdict
        :param db: pytable database file
        """
        self.params = params

        self.N = params.grid_n
        self.M = params.grid_m
        self.grid_shape = [self.N, self.M]
        self.moves = params.moves
        self.observe_directions = params.observe_directions #定义观测的方向

        self.num_action = params.num_action #定义动作的数量
        self.num_obs = params.num_obs #定义观测的数量
        self.obs_len = len(self.observe_directions) #观测的长度
        self.num_state = self.N * self.M #定义状态的数量

        self.grid = None


    def generate_trajectories(self, path, num_traj):
        #生成环境和轨迹，db是数据文件
        params = self.params
        max_traj_len = params.traj_limit
        res_b0 = []
        res_path = []
        res_expR = []
        res_start_state = []
        res_goal_state = []
        traj_i = 0
        while(traj_i<num_traj):
            # generate a QMDP object, initial belief, initial state and goal state
            # also generates a random grid for the first iteration
            qmdp, b0, start_state, goal_states = self.random_instance(generate_grid=(traj_i == 0))
            #QMDP中传入了T、R、Z
            qmdp.solve() #QMDP模型进行求解，Planner进行规划完毕

            state = start_state
            b = b0.copy()  # linear belief
            reward_sum = 0.0  # accumulated reward
            gamma_acc = 1.0

            beliefs = [] # includes start and goal
            states = [] # includes start and goal
            actions = [] # first action is always stay. Excludes action after reaching goal
            observs = [] # Includes observation at start but excludes observation after reaching goal

            collisions = 0
            failed = False
            step_i = 0

            while True:
                beliefs.append(b)
                states.append(state)

                # finish if state is terminal, i.e. we reached a goal state
                if all([np.isclose(qmdp.T[x][state, state], 1.0) for x in range(params.num_action)]):
                    assert state in goal_states
                    break

                # stop if trajectory limit reached
                if step_i >= max_traj_len:  # it should reach terminal state sooner or later
                    failed = True
                    break

                # choose action
                if step_i == 0:
                    # dummy first action
                    act = params.stayaction
                else:
                    act = qmdp.qmdp_action(b) #用b和之前solve()方法算好的Q，获得动作act

                # simulate action
                state, r = qmdp.transition(state, act) #执行动作，得到下一个状态snext和奖励r
                bprime, obs, b = qmdp.belief_update(b, act, state_after_transition=state) #得到bprime，观测obs和b

                actions.append(act)
                observs.append(obs)

                reward_sum += r * gamma_acc
                gamma_acc = gamma_acc * qmdp.discount

                # count collisions
                if np.isclose(r, params.R_obst):
                    collisions += 1

                step_i += 1

            # add to database
            if failed:
                continue
            traj_i += 1
            traj_len = step_i

            # step: state (linear), action, observation (linear)
            step = np.stack([states[:traj_len], actions[:traj_len], observs[:traj_len]], axis=1)#将每个state,action,observation组合得到一个序列

            # sample: env_id, goal_state, step_id, traj_length, collisions, failed
            # length includes both start and goal (so one step path is length 2)

            res_path.append(step)
            res_b0.append(beliefs[:1])
            res_goal_state.append(goal_states[0])
            res_start_state.append(start_state)
            res_expR.append(reward_sum)
        # add environment only after adding all trajectories
        return self.grid[None],res_path,res_b0,res_goal_state,res_start_state,res_expR

    def random_instance(self, generate_grid=True):
        """
        Generate a random problem instance for a grid.
        Picks a random initial belief, initial state and goal states.
        :param generate_grid: generate a new grid and pomdp model if True, otherwise use self.grid
        :return: #返回QMDP模型、初始置信度b0，开始状态和结束状态
        """
        while True:
            if generate_grid:
                #随机生成格子，随机初始化障碍
                self.grid = self.random_grid(self.params.grid_n, self.params.grid_m, self.params.Pobst)
                self.gen_pomdp()  # generates pomdp model, self.T, self.Z, self.R

            while True:
                # sample initial belief, start, goal
                b0, start_state, goal_state = self.gen_start_and_goal()
                if b0 is None:
                    assert generate_grid
                    break  # regenerate obstacles

                goal_states = [goal_state]

                # reject if start == goal
                if start_state in goal_states:
                    continue

                # create qmdp                                                                    2021-03-21结束标记
                qmdp = self.get_qmdp(goal_states)  # makes soft copies from self.T{R,Z}simple
                # it will also convert to csr sparse, and set qmdp.issparse=True

                return qmdp, b0, start_state, goal_states #返回QMDP模型、初始置信度b0，开始状态和结束状态

    def gen_pomdp(self):
        # construct all POMDP model(R, T, Z)建立一个POMDP模型，有R,T,Z
        self.Z = self.build_Z()
        self.T, Tml, self.R = self.build_TR()

        # transform into graph with opposite directional actions, so we can compute path from goal
        G = {i: {} for i in range(self.num_state)}
        for a in range(self.num_action):
            for s in range(self.num_state):
                snext = Tml[s, a]
                if s != snext:
                    G[snext][s] = 1  # edge with distance 1
        self.graph = G #建立状态间的可达矩阵

    def build_Z(self):
        params = self.params

        Pobs_succ = params.Pobs_succ #观测成功率

        Z = np.zeros([self.num_action, self.num_state, self.num_obs], 'f')#Z表示为A_size*S_size*O_size

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j]) #将数组转为矩阵，每个位置都检测
                state = self.state_bin_to_lin(state_coord) #获得下标索引在grid_shape中的索引

                # first build observation
                obs = np.zeros([self.obs_len])  # 1 or 0 in four directions
                for direction in range(self.obs_len):
                    neighb = self.apply_move(state_coord, np.array(self.observe_directions[direction]))
                    #执行动作，获得要观测的邻居
                    if self.check_free(neighb):#check_free检测了当前状态索引是否在合理范围内，并且是否是障碍点
                        obs[direction] = 0
                    else:
                        obs[direction] = 1

                # add all observations with their probabilities
                for obs_i in range(self.num_obs):#计算所有观测状态的可能性
                    dist = np.abs(self.obs_lin_to_bin(obs_i) - obs).sum()#相似度，当前观测与观测i的距离
                    prob = np.power(1.0 - Pobs_succ, dist) * np.power(Pobs_succ, self.obs_len - dist)#求
                    Z[:, state, obs_i] = prob #当前状态获得观测o的先验概率，对于每个动作都一样

                # sanity check 检查一下obs_i的概率是否为1。因为这里的式子等价于二项式展开（1 - Pobs_succ + Pobs_succ）^obs_len
                assert np.isclose(1.0, Z[0, state, :].sum())

        return Z

    def build_TR(self):
        """
        Builds transition (T) and reward (R) model for a grid.
        The model does not capture goal states, which must be incorporated later.
        :return: transition model T, maximum likely transitions Tml, reward model R
        """
        params = self.params
        Pmove_succ = params.Pmove_succ

        # T, R does not capture goal state, it must be incorporated later
        T = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        R = [scipy.sparse.lil_matrix((self.num_state, self.num_state), dtype='f')
             for x in range(self.num_action)]  # probability of transition with a0 from s1 to s2
        # goal will be defined as a terminal state, all actions remain in goal with 0 reward

        # maximum likely versions
        Tml = np.zeros([self.num_state, self.num_action], 'i')  # Tml[s, a] --> next state
        Rml = np.zeros([self.num_state, self.num_action], 'f')  # Rml[s, a] --> reward after executing a in s

        for i in range(self.N):
            for j in range(self.M):
                state_coord = np.array([i, j])
                state = self.state_bin_to_lin(state_coord)

                # build T and R
                for act in range(self.num_action):
                    neighbor_coord = self.apply_move(state_coord, np.array(self.moves[act]))
                    if self.check_free(neighbor_coord):
                        Rml[state, act] = params['R_step'][act]
                    else:
                        neighbor_coord[:2] = [i, j]  # dont move if obstacle or edge of world
                        # alternative: neighbor_coord = state_coord
                        Rml[state, act] = params['R_obst']

                    neighbor = self.state_bin_to_lin(neighbor_coord)
                    Tml[state, act] = neighbor
                    if state == neighbor:
                        # shortcut if didnt move
                        R[act][state, state] = Rml[state, act]
                        T[act][state, state] = 1.0
                    else:
                        R[act][state, state] = params['R_step'][act]
                        # cost if transition fails (might be lucky and avoid wall)
                        R[act][state, neighbor] = Rml[state, act]
                        T[act][state, state] = 1.0 - Pmove_succ
                        T[act][state, neighbor] = Pmove_succ

        return T, Tml, R

    def gen_start_and_goal(self, maxtrials=1000):
        #返回最初的信念、开始状态、目标状态
        """
        Pick an initial belief, initial state and goal state randomly
        """
        free_states = np.nonzero((self.grid == FREESTATE).flatten())[0]#返回空位置的索引
        freespace_size = len(free_states)

        for trial in range(maxtrials):
            b0sizes = np.floor(freespace_size / np.power(2.0, np.arange(20)))
            b0sizes = b0sizes[:np.nonzero(b0sizes < 1)[0][0]]
            b0size = int(np.random.choice(b0sizes))

            b0ind = np.random.choice(free_states, b0size, replace=False)#从free_states中选取b0size个样本，并且每个样本只能选择1次
            b0 = np.zeros([self.num_state])
            b0[b0ind] = 1.0 / b0size  # uniform distribution over sampled states，b0size个b0ind点都有1/b0size的概率

            # sanity check
            for state in b0ind:
                coord = self.state_lin_to_bin(state)
                assert self.check_free(coord)

            # sample initial state from initial belief  从置信度为1/b0size的点中选取一个起始点
            start_state = np.random.choice(self.num_state, p=b0)

            # sample goal uniformly from free space 从free_states的点中，随机选取一个终点
            goal_state = np.random.choice(free_states)

            # check if path exists from start to goal, if not, pick a new set
            D, path_pointers = dijkstra.Dijkstra(self.graph, goal_state)  # map of distances and predecessors
            # 用Dijkstra算法得到可达的点对应的距离D(如：{'y': 7, 'x': 5, 's': 0, 'u': 8, 'v': 9})，和经过的点path_pointers(如：{'y': 'x', 'x': 's', 'u': 'x', 'v': 'u'})
            if start_state in D:
                break
        else:
            # never succeeded
            raise ValueError

        return b0, start_state, goal_state

    def get_qmdp(self, goal_states):
        qmdp = QMDP(self.params)#初始化QMDP模型

        qmdp.processT(self.T)  # this will make a hard copy
        qmdp.processR(self.R)
        qmdp.processZ(self.Z) #传入T、R、Z参数，并进行硬拷贝

        qmdp.set_terminals(goal_states, reward=self.params.R_goal)#传入goal和奖励参数，对于建立奖励函数和goal点的转移函数

        qmdp.transfer_all_sparse() #把T、R、Z变成稀疏的
        return qmdp

    @staticmethod
    def sample_free_state(map):
        """
        Return the coordinates of a random free state from the 2D input map
        """
        while True:
            coord = [random.randrange(map.shape[0]), random.randrange(map.shape[1])]
            if map[coord[0],coord[1],0] == FREESTATE:
                return coord

    @staticmethod
    def outofbounds(map, coord):
        return (coord[0] < 0 or coord[0] >= map.shape[0] or coord[1] < 0 or coord[1] >= map.shape[1])

    @staticmethod
    def apply_move(coord_in, move):
        coord = coord_in.copy()
        coord[:2] += move[:2] #例如[0,0]+[0,1]
        return coord

    def check_free(self, coord):
        return (not GridBase.outofbounds(self.grid, coord) and self.grid[coord[0], coord[1]] != OBSTACLE)

    @staticmethod
    def random_grid(N, M, Pobst):
        #随机生成格子世界
        grid = np.zeros([N, M])

        # borders，格子的四周是障碍
        grid[0, :] = OBSTACLE
        grid[-1, :] = OBSTACLE
        grid[:, 0] = OBSTACLE
        grid[:, -1] = OBSTACLE

        rand_field = np.random.rand(N, M)#在格子世界中随机生成概率
        grid = np.array(np.logical_or(grid, (rand_field < Pobst)), 'i')#如果概率<0.25则设置为障碍，并且原有边界障碍不变
        return grid

    def obs_lin_to_bin(self, obs_lin):
        obs = np.array(np.unravel_index(obs_lin, [2,2,2,2]), 'i')
        if obs.ndim > 2:
            raise NotImplementedError
        elif obs.ndim > 1:
            obs = np.transpose(obs, [1,0])
        return obs

    def obs_bin_to_lin(self, obs_bin):
        return np.ravel_multi_index(obs_bin, [2,2,2,2])

    def state_lin_to_bin(self, state_lin):
        return np.unravel_index(state_lin, self.grid_shape)

    def state_bin_to_lin(self, state_coord):
        return np.ravel_multi_index(state_coord, self.grid_shape)


    def process_goals(self, goal_state):
        """
        :param goal_state: linear goal state
        :return: goal image, same size as grid
        """
        goal_img = np.zeros([goal_state.shape[0], self.N, self.M], 'i')
        goalidx = np.unravel_index(goal_state, [self.N, self.M])

        goal_img[np.arange(goal_state.shape[0]), goalidx[0], goalidx[1]] = 1

        return goal_img

    def process_beliefs(self, linear_belief):
        """
        :param linear_belief: belief in linear space
        :return: belief reshaped to grid size
        """
        batch = (linear_belief.shape[0] if linear_belief.ndim > 1 else 1)
        b = linear_belief.reshape([batch, self.params.grid_n, self.params.grid_m, ])
        if b.dtype != np.float:
            return b.astype('f')

        return b


def generate_grid_data(path, N=30, M=30, num_env=10000, traj_per_env=5, Pmove_succ=1.0, Pobs_succ=1.0):
    """
    :param path: path for data file. use separate folders for training and test data
    :param N: grid rows
    :param M: grid columnts
    :param num_env: number of environments in the dataset (grids)
    :param traj_per_env: number of trajectories per environment (different initial state, goal, initial belief)
    :param Pmove_succ: probability of transition succeeding, otherwise stays in place
    :param Pobs_succ: probability of correct observation, independent in each direction
    """

    params = dotdict({
        'grid_n': N,
        'grid_m': M,
        'Pobst': 0.25,  # probability of obstacles in random grid

        'R_obst': -10, 'R_goal': 20, 'R_step': -0.1, #障碍的奖励-10，目标奖励20，空余步骤奖励-0.1
        'discount': 0.99, #衰减0.99
        'Pmove_succ':Pmove_succ,#移动成功概率
        'Pobs_succ': Pobs_succ, #观测成功概率

        'num_action': 5, #动作数量5，UVIN是8（不包括stay，添加了左上、右上、左下、右下）
        'moves': [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]],  # right, down, left, up, stay
        'stayaction': 4, #呆在原地的动作索引是4

        'num_obs': 16, #观测的数量2^4,UVIN里面应该观测有256（2^8）种
        'observe_directions': [[0, 1], [1, 0], [0, -1], [-1, 0]],#观测方向上下左右,UVIN中新加左上右上左下右下
        })

    params['obs_len'] = len(params['observe_directions'])
    params['num_state'] = params['grid_n']*params['grid_m']
    params['traj_limit'] = 4 * (params['grid_n'] + params['grid_m'])#轨迹长度限制
    params['R_step'] = [params['R_step']] * params['num_action']


    # randomize seeds, set to previous value to determinize random numbers
    np.random.seed()
    random.seed()

    # grid domain object
    domain = GridBase(params)

    # make database file
    Map = []
    Path = []
    B0 = []
    Goal = []
    Start = []
    ExpReward =[]
    for env_i in range(num_env):
        print ("Generating env %d with %d trajectories "%(env_i, traj_per_env))
        map,res_path, res_b0, res_goal_state, res_start_state, res_expR = domain.generate_trajectories(path, num_traj=traj_per_env)#生成环境和轨迹
        Map.extend(map)
        Path.extend(res_path)
        B0.extend(res_b0)
        Goal.extend(res_goal_state)
        Start.extend(res_start_state)
        ExpReward.extend(res_expR)
    torch.save({'Map': Map, 'B0': B0, 'Path':Path, 'goal': Goal, 'Start': Start},path)
    print ("Done.")


def main():

    # training data
    generate_grid_data(path='data/regular_maze/train',N=10, M=10, num_env=10000, traj_per_env=5,
                     Pmove_succ=1.0, Pobs_succ=1.0)

    # test data
    generate_grid_data(path='data/regular_maze/test',N=10, M=10, num_env=500, traj_per_env=1,
                       Pmove_succ=1.0, Pobs_succ=1.0)


# default
if __name__ == "__main__":
    main()