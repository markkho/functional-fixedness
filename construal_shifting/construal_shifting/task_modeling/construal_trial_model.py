import numpy as np
import random

from msdm.core.utils.funcutils import cached_property
from msdm.core.distributions import DictDistribution
from msdm.core.utils.funcutils import method_cache
from msdm.algorithms import PolicyIteration
from msdm.core.mdp.tables import StateTable

from construal_shifting.construal import maze_construals, construal_size
from construal_shifting.task_modeling.cached_gridworld import GridWorld

from msdm.core.mdp import  TabularPolicy

import typing
if typing.TYPE_CHECKING:
    from construal_shifting.task_modeling.simulated_participant_model import SimulatedGridNavigationTrialData

class ConstrualTrialModel:
    # this logic makes it so we only create a single 
    # object for each set of parameters
    __cache__ = {}
    @staticmethod
    def instance_key(*args, **kws):
        return (args, tuple(kws.items()))
    def __new__(cls, *args, **kws):
        if cls.instance_key(*args, **kws) not in cls.__cache__:
            return super().__new__(cls)
        else:
             return cls.__cache__[cls.instance_key(*args, **kws)]
        
    def __init__(
        self,
        **trial_params
    ):
        if self.instance_key(**trial_params) in self.__cache__:
            return
        else:
            self.__cache__[self.instance_key(**trial_params)] = self
        self.trial_params = trial_params
    @cached_property
    def true_mdp(self) -> GridWorld:
        return GridWorld(**self.trial_params)
    @cached_property
    def construal_tile_arrays(self):
        return tuple(maze_construals('\n'.join(self.trial_params['tile_array'])))
    @cached_property
    def cognitive_cost_vec(self):
        vec = np.array([construal_size(c) for c in self.construal_tile_arrays])
        vec.setflags(write=False)
        return vec
    @cached_property
    def construed_mdps(self):
        mdps = []
        for c in self.construal_tile_arrays:
            mdps.append(GridWorld(**{**self.trial_params, 'tile_array': c.split('\n')}))
        return tuple(mdps)
    @cached_property
    def _computed_plans_results(self):
        return PolicyIteration(undefined_value=0.).batch_plan_on(self.construed_mdps)
    @cached_property
    def computed_plan_values(self):
        return tuple([res.action_value for res in self._computed_plans_results])
    @cached_property
    def computed_plan_values_matrix(self):
        matrix = np.stack([np.array(action_value) for action_value in self.computed_plan_values])
        matrix.setflags(write=False)
        return matrix
    @cached_property
    def computed_plan_advantage_matrix(self):
        matrix = \
            self.computed_plan_values_matrix -\
            self.computed_plan_values_matrix.max(-1, keepdims=True)
        matrix.setflags(write=False)
        return matrix
    @cached_property
    def computed_plans(self):
        return tuple([res.policy for res in self._computed_plans_results])
    @cached_property
    def computed_plans_matrix(self):
        matrix = np.stack([np.array(plan) for plan in self.computed_plans])
        matrix.setflags(write=False)
        return matrix
    @cached_property
    def behavioral_utilities_vec(self):
        vec = np.array([plan.evaluate_on(self.true_mdp).initial_value for plan in self.computed_plans])
        vec.setflags(write=False)
        return vec
    @method_cache
    def construal_dist(self, construal_cost_weight=1.0, construal_inverse_temp=1.0):
        probs = self.construal_dist_vec(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp
        )
        return DictDistribution(zip(self.construal_tile_arrays, probs))
    @method_cache
    def construal_dist_vec(self, construal_cost_weight=1.0, construal_inverse_temp=1.0):
        vals = self.behavioral_utilities_vec - construal_cost_weight*self.cognitive_cost_vec
        vals *= construal_inverse_temp
        vals[:] = vals - vals.max()
        probs = np.exp(vals)
        probs = probs/probs.sum()
        probs.setflags(write=False)
        return probs
    def construal_set_value(
        self,
        construal_cost_weight=1.0,
        construal_inverse_temp=1.0,
        construal_prior=None,
    ):
        construal_values = self.behavioral_utilities_vec - construal_cost_weight*self.cognitive_cost_vec
        construal_policy = self._construal_posterior(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=construal_prior,
        )
        if (construal_policy == 0.).all():
            return float('-inf')
        construal_values[construal_policy == 0.] = 0
        return (construal_values*construal_policy).sum()
    def construal_set_cognitive_cost(
        self,
        construal_cost_weight=1.0,
        construal_inverse_temp=1.0,
        construal_prior=None,
    ):
        construal_policy = self._construal_posterior(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=construal_prior,
        )
        construal_costs = construal_cost_weight*self.cognitive_cost_vec
        construal_costs[construal_policy == 0.] = 0
        return (construal_costs*construal_policy).sum()
    @method_cache
    def _construal_prior_vec(self, construal_prior):
        vec = np.array([construal_prior(c) for c in self.construal_tile_arrays])
        vec.setflags(write=False)
        return vec
    def _construal_posterior(
        self,
        construal_cost_weight=1.0,
        construal_inverse_temp=1.0,
        construal_prior=None,
    ):
        construal_dist_vec = self.construal_dist_vec(
            construal_cost_weight=construal_cost_weight, 
            construal_inverse_temp=construal_inverse_temp
        )
        if construal_prior is not None:
            construal_prior_vec = self._construal_prior_vec(construal_prior)
            construal_dist_vec = construal_prior_vec*construal_dist_vec
            if construal_dist_vec.sum() > 0:
                construal_dist_vec = construal_dist_vec/construal_dist_vec.sum()
        return construal_dist_vec
    def expected_location_walls(
        self,
        construal_cost_weight=1.0,
        construal_inverse_temp=1.0,
        construal_prior=None
    ):
        construal_dist_vec = self._construal_posterior(
            construal_cost_weight=construal_cost_weight, 
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=construal_prior,
        )
        cdist = dict(zip(self.construal_tile_arrays, construal_dist_vec))
        exp_walls = np.zeros(len(self.true_mdp.state_list))
        for construal, mdp in zip(self.construal_tile_arrays, self.construed_mdps):
            exp_walls += mdp.wall_state_vec*cdist[construal]
        return StateTable.from_state_list(
            state_list=self.true_mdp.state_list,
            data=exp_walls
        )
    @method_cache
    def epsilon_softmax_policies(
        self,
        action_inverse_temp=float('inf'),
        action_random_choice=0.0
    ) -> np.ndarray:
        if action_inverse_temp == float('inf'):
            computed_plans = self.computed_plans_matrix
        else:
            computed_plans = self.computed_plan_advantage_matrix*action_inverse_temp
            assert (computed_plans <= 0).all(), "values need to all be negative to avoid overflow"
            np.exp(computed_plans, out=computed_plans)
            np.divide(computed_plans, computed_plans.sum(-1, keepdims=True), out=computed_plans)
        if action_random_choice == 0.0:
            return computed_plans
        rand_policy = np.ones(computed_plans.shape[-1])/computed_plans.shape[-1]
        computed_plans = \
            computed_plans*(1 - action_random_choice) + \
            rand_policy[None, None, :]*action_random_choice
        return computed_plans
    def trajectory_prob(
        self, 
        states,
        actions,
        construal_cost_weight=1.0,
        construal_inverse_temp=1.0,
        construal_prior=None,
        action_inverse_temp=10.0,
        action_random_choice=0.05
    ):
        assert len(states) == len(actions)
        construal_dist_vec = self._construal_posterior(
            construal_cost_weight=construal_cost_weight, 
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=construal_prior,
        )
        
        computed_plans = self.epsilon_softmax_policies(
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice
        )
        
        traj_probs = np.ones(len(self.construal_tile_arrays))
        for s, a in zip(states, actions):
            si = self.true_mdp.state_list.index(s)
            ai = self.true_mdp.action_list.index(a)
            traj_probs *= computed_plans[:, si, ai]
        return (traj_probs*construal_dist_vec).sum()
    
    def clear_method_caches(self):
        self._cache_construal_dist_vec.clear()
        self._cache_construal_dist_vec.clear()
        self._cache_epsilon_softmax_policies.clear()

    @classmethod
    def clear_instance_method_caches(cls):
        for inst in cls.__cache__.values():
            inst.clear_method_caches()

    def simulate_trial(
        self,
        last_cset,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice,
        construal_set_stickiness,
        trial_index = None,
        grid_name = None,
        rng : random.Random = random,
    ) -> "SimulatedGridNavigationTrialData":
        # avoid circular import
        from construal_shifting.task_modeling.participant_model import coarse_identifier, fine_identifier
        from construal_shifting.task_modeling.simulated_participant_model import SimulatedGridNavigationTrialData

        # also get cset values
        coarse_val = self.construal_set_value(
            construal_prior=coarse_identifier,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
        )
        fine_val = self.construal_set_value(
            construal_prior=fine_identifier,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
        )
        set_values = np.array([coarse_val, fine_val])

        # sample a next construal set based on intrinsic and stickiness value
        set_nextset_switch_cost = (1 - np.eye(2))*construal_set_stickiness
        set_nextset_values = set_values[None, :] - set_nextset_switch_cost
        set_nextset_values = np.exp(set_nextset_values)
        set_nextset_values /= set_nextset_values.sum(-1, keepdims=True)
        cset_tf = set_nextset_values[['coarse', 'fine'].index(last_cset)]
        cset = rng.choices(['coarse', 'fine'], weights=cset_tf)[0]

        # sample a construal and computed plan
        c_probs = self._construal_posterior(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            construal_prior={'fine': fine_identifier, 'coarse': coarse_identifier}[cset],
        )
        c_i = rng.choices(range(len(c_probs)), weights=c_probs)[0]
        computed_plans = self.epsilon_softmax_policies(
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice
        )
        cpolicy = TabularPolicy.from_state_action_lists(
            state_list=self.true_mdp.state_list,
            action_list=self.true_mdp.action_list,
            data=computed_plans[c_i]
        )
        sim_res = cpolicy.run_on(self.true_mdp, rng=rng)
        coarse_cognitive_cost = self.construal_set_cognitive_cost(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=coarse_identifier
        )
        fine_cognitive_cost = self.construal_set_cognitive_cost(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            construal_prior=fine_identifier
        )
        state_traj = [(x, y) for x, y in sim_res.state]
        assert self.true_mdp.is_absorbing(state_traj[-1])
        action_traj = [a for a in sim_res.action if a is not None]
        return SimulatedGridNavigationTrialData(
            trial_params=self.trial_params,
            coarse_set_cognitive_cost=coarse_cognitive_cost,
            fine_set_cognitive_cost=fine_cognitive_cost,
            state_traj=state_traj,
            action_traj=action_traj,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice,
            construal_set_stickiness=construal_set_stickiness,
            current_construal_set=cset,
            trial_index=trial_index,
            grid_name=grid_name,
        )