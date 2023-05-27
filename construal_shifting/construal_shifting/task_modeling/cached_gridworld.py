from dataclasses import dataclass
from itertools import product
import numpy as np

from msdm.domains.gridmdp import GridMDP
from msdm.core.table.tableindex import domaintuple
from msdm.core.distributions import DictDistribution
from msdm.core.mdp.tables import StateTable
from msdm.core.utils.funcutils import cached_property

MAIN_OBSTACLE_COLOR = (173/255, 216/255, 230/255, 1.)
BROKEN_OBSTACLE_COLOR = (173/255, 216/255, 230/255, .5)

class GridWorld(GridMDP):
    _templates = {}
    def __init__(
        self,
        tile_array,
        absorbing_features=("g",),
        wall_features=("#",),
        default_features=(".",),
        initial_features=("s",),
        step_cost=-1,
        wall_bump_cost=0,
        success_prob=1.0,
        discount_rate=1.0,
        wait_action=False,
    ):
        super().__init__('\n'.join(tile_array))
        self.tile_array = np.array([list(row) for row in tile_array[::-1]])
        self.wall_features = wall_features
        self.initial_features = initial_features
        self.absorbing_features = absorbing_features
        self.step_cost = step_cost
        self.wall_bump_cost = wall_bump_cost
        self.success_prob = success_prob
        self.discount_rate = discount_rate
        self.template = self.__class__.get_template(
            height=self.tile_array.shape[0],
            width=self.tile_array.shape[1],
            step_cost=step_cost,
            wait_action=wait_action,
        )
    
    def plot(self, feature_colors=None, feature_markers=None, ax=None):
        if feature_colors is None:
            feature_colors = {
                **{f: MAIN_OBSTACLE_COLOR for f in "ABCDEFGHIJK"},
                **{f: BROKEN_OBSTACLE_COLOR for f in "abcdefghijk"},
                '#': 'k',
                '.': 'w',
                '$': 'yellow'
            }
        if feature_markers is None:
            feature_markers = {'@': 'o', '$': 'x'}
        gwp = super().plot(
            feature_colors=feature_colors,
            feature_markers=feature_markers,
            ax=ax
        )
        return gwp
        
    @cached_property
    def state_list(self):
        return self.template.state_list
    @cached_property
    def action_list(self):
        return self.template.action_list
    @cached_property
    def transition_matrix(self):
        return self.transition_reward_matrices[0]
    @cached_property
    def reward_matrix(self):
        return self.transition_reward_matrices[1]
    @cached_property
    def transition_reward_matrices(self):
        transition_matrix = self.template.transition_matrix.copy()
        reward_matrix = self.template.reward_matrix.copy()
        
        # wall transitions
        transition_matrix[:, :, self.wall_state_vec] = 0
        pre_hit_wall_state, pre_hit_wall_action = (transition_matrix.sum(-1) < 1).nonzero()
        transition_matrix[pre_hit_wall_state, pre_hit_wall_action, pre_hit_wall_state] = 1
        reward_matrix[pre_hit_wall_state, pre_hit_wall_action, :] += self.wall_bump_cost
        
        # stochastic movement
        transition_matrix = \
            self.template.eye[:, None, :]*(1 - self.success_prob) + transition_matrix*self.success_prob
        
        # absorbing transitions
        transition_matrix[self.absorbing_state_vec, :, :] = 0
        transition_matrix[self.absorbing_state_vec, :, self.absorbing_state_vec] = 1
        reward_matrix[self.absorbing_state_vec, :, :] = 0
        
        transition_matrix.setflags(write=False)
        reward_matrix.setflags(write=False)
        return transition_matrix, reward_matrix
    @cached_property
    def location_feature_table(self):
        return StateTable.from_state_list(
            state_list=self.state_list,
            data=self.location_feature_vec,
        )
    @cached_property
    def location_feature_vec(self):
        vec = np.array([self.location_feature_dict[loc] for loc in self.state_list])
        vec.setflags(write=False)
        return vec
    @cached_property
    def action_matrix(self):
        return self.template.action_matrix
    @cached_property
    def absorbing_state_vec(self):
        absorbing_state_vec = np.isin(self.location_feature_vec, tuple(self.absorbing_features))
        absorbing_state_vec.setflags(write=False)
        return absorbing_state_vec
    @cached_property
    def wall_state_vec(self):
        wall_state_vec = np.isin(self.location_feature_vec, tuple(self.wall_features))
        wall_state_vec.setflags(write=False)
        return wall_state_vec
    @cached_property
    def initial_state_vec(self):
        initial_state_vec = np.isin(self.location_feature_vec, tuple(self.initial_features))
        initial_state_vec = initial_state_vec/initial_state_vec.sum()
        initial_state_vec.setflags(write=False)
        return initial_state_vec
    @classmethod
    def get_template(cls, height, width, step_cost, wait_action):
        if wait_action:
            actions = domaintuple([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
        else:
            actions = domaintuple([(0, 1), (0, -1), (1, 0), (-1, 0)])
        key = (height, width, step_cost, actions)
        try:
            template = cls._templates[key]
        except KeyError:
            cls._templates[key] = GridWorldTemplate(
                height=height,
                width=width,
                step_cost=-1,
                actions=actions,
            )
            template = cls._templates[key]
        return template
    
    def initial_state_dist(self):
        nonzero = (self.initial_state_vec > 0).nonzero()
        return DictDistribution(zip(
            [self.state_list[si[0]] for si in nonzero],
            self.initial_state_vec[nonzero]
        ))
    def is_absorbing(self, s):
        return self.absorbing_state_vec[self.state_list.index(s)]
    def next_state_dist(self, s, a):
        ns_dist = self.transition_matrix[
            self.state_list.index(s), 
            self.action_list.index(a), 
        ]
        nonzero = (ns_dist > 0).nonzero()
        return DictDistribution(zip([self.state_list[si[0]] for si in nonzero], ns_dist[nonzero]))
    def reward(self, s, a, ns):
        return self.reward_matrix[
            self.state_list.index(s), 
            self.action_list.index(a), 
            self.state_list.index(ns), 
        ]
    def actions(self, s):
        return self.action_list

class GridWorldTemplate:
    def __init__(
        self,
        height,
        width,
        step_cost=-1,
        actions = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)),
    ):
        state_list = domaintuple(product(range(width), range(height)))
        action_list = domaintuple(actions)
        transition_matrix = np.zeros((height*width, len(actions), height*width))
        for si, s in enumerate(state_list):
            for ai, a in enumerate(actions):
                ns = (s[0]+a[0], s[1]+a[1])
                if ns[0] < 0 or ns[0] >= width or ns[1] < 0 or ns[1] >= height:
                    ns = s
                nsi = state_list.index(ns)
                transition_matrix[si, ai, nsi] = 1
        transition_matrix.setflags(write=False)
        reward_matrix = np.ones((height*width, len(actions), height*width))*step_cost
        reward_matrix.setflags(write=False)
        action_matrix = np.ones((height*width, len(actions),), dtype=bool)
        action_matrix.setflags(write=False)
        
        self.action_list = action_list
        self.state_list = state_list
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.action_matrix = action_matrix
        self.eye = np.eye(len(state_list))
