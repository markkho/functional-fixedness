from itertools import product
from frozendict import frozendict
from msdm.core.utils.gridstringutils import  string_to_element_array
from msdm.domains import GridWorld
from msdm.domains.gridworld.mdp import TERMINALSTATE
from msdm.core.mdp import TabularMarkovDecisionProcess
from msdm.core.distributions import DictDistribution
from msdm.algorithms import PolicyIteration

import numpy as np
class GridWorld2(GridWorld, TabularMarkovDecisionProcess):
    def __init__(
        self,
        tile_array,
        feature_rewards=None,
        absorbing_features=("g",),
        wall_features=("#",),
        default_features=(".",),
        initial_features=("s",),
        step_cost=-1,
        wall_bump_cost=0,
        success_prob=1.0,
        discount_rate=1.0
    ):
        """An optimized gridworld implementation """

        self.discount_rate = discount_rate

        if feature_rewards is None:
            feature_rewards = {}

        h = len(tile_array)
        w = len(tile_array[0])
        self._height = h
        self._width = w
        self.tile_array = tile_array
        ss = [frozendict(x=x, y=y) for x, y in product(range(w), range(h))] + [TERMINALSTATE, ]
        self._state_list = ss
        aa = [
            frozendict({'dx': 0, 'dy': 0}),
            frozendict({'dx': 1, 'dy': 0}),
            frozendict({'dx': -1, 'dy': 0}),
            frozendict({'dy': 1, 'dx': 0}),
            frozendict({'dy': -1, 'dx': 0})
        ]
        self._action_list = aa

        obs = []
        absorbing = []
        self._absorbingStates = []
        self._walls = []
        self._initStates = []
        self._locFeatures = {}
        state_reward = []
        initial_states = []
        for x in range(w):
            for y in range(h):
                s = frozendict(x=x, y=y)
                y_ = h - y - 1
                f = tile_array[y_][x]
                self._locFeatures[s] = f
                if f in wall_features:
                    obs.append(1)
                    self._walls.append(s)
                else:
                    obs.append(0)
                if f in absorbing_features:
                    absorbing.append(1)
                    self._absorbingStates.append(s)
                else:
                    absorbing.append(0)
                state_reward.append(feature_rewards.get(f, 0.0) + step_cost)
                if f in initial_features:
                    initial_states.append(1)
                    self._initStates.append(s)
                else:
                    initial_states.append(0)
        obs = np.array(obs + [0, ])
        absorbing = np.array(absorbing + [0, ])
        
        state_reward = np.array(state_reward)
        s0 = np.array(initial_states + [0,])
        s0 = s0/s0.sum()
        self._initial_state_vec = s0

        tf = np.zeros((w*h + 1, len(aa), w*h + 1))
        rf = np.zeros((w*h + 1, len(aa), w*h + 1))
        am = np.zeros((w*h + 1))
        for x, y in product(range(w), range(h)):
            s = frozendict(x=x, y=y)
            si = x*h + y
            if absorbing[si]:
                tf[x*h + y, :, -1] = 1
                continue
            for ai, a in enumerate(aa):
                nx, ny = x + a['dx'], y + a['dy']
                nsi = nx*h + ny
                if (0 <= nx < w) and (0 <= ny < h) and (not obs[nsi]):
                    tf[si, ai, nsi] = 1
                    rf[si, ai, nsi] = state_reward[nsi]
                else:
                    tf[si, ai, si] += 1
                    rf[si, ai, si] = state_reward[si] + wall_bump_cost
        tf[-1, :, -1] = 1 #terminal state
        self._transition_matrix = tf
        self._reward_matrix = rf
        nt = np.ones(len(ss))
        nt[-1] = 0
        self._nonterminal_state_vec = nt
        
        # reachability
        mp = tf.sum(axis=1)
        mp /= mp.sum(axis=1, keepdims=True)
        mp = np.nan_to_num(mp, copy=False)
        oc = np.linalg.inv(np.eye(mp.shape[0]) - .9*mp*nt[:,None])
        rs = np.einsum("sn,s->n", oc, s0) > 0
        self._reachable_state_vec = rs
    
    @property
    def state_list(self):
        return self._state_list
    @property
    def action_list(self):
        return self._action_list
    @property
    def initial_state_vec(self):
        return self._initial_state_vec
    @property
    def transition_matrix(self):
        return self._transition_matrix
    @property
    def reward_matrix(self):
        return self._reward_matrix
    @property
    def nonterminal_state_vec(self):
        return self._nonterminal_state_vec
    @property
    def reachable_state_vec(self):
        return self._reachable_state_vec
    
    def next_state_dist(self, s, a):
        nsprobs = self.transition_matrix[self.state_list.index(s), self.action_list.index(a), :]
        return DictDistribution(zip(self.state_list, nsprobs))
        
    def initial_state_dist(self):
        return DictDistribution(zip(self.state_list, self.initial_state_vec))
    
    def reward(self, s, a, ns):
        return self.reward_matrix[self.state_list.index(s), self.action_list.index(a), self.state_list.index(ns)]
    
    def actions(self, s):
        return self.action_list
    
    def is_terminal(self, s):
        return s == TERMINALSTATE