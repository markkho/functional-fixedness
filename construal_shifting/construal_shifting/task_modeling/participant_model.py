import warnings 
import numpy as np
from itertools import product

from msdm.core.utils.funcutils import cached_property, method_cache
from msdm.core.table import Table, TableIndex
from msdm.core.table.tableindex import Field
from msdm.core.table.tableindex import domaintuple

from construal_shifting.construal import construal_level
from construal_shifting.task_modeling.construal_trial_model import ConstrualTrialModel
from construal_shifting.task_modeling.base_dataclasses import ParticipantDataBase, GridNavigationTrialDataBase

def coarse_identifier(c):
    return construal_level(c) == 'coarse'
def fine_identifier(c):
    return construal_level(c) == 'fine'

class ParticipantModel:
    default_gw_params = dict(
        initial_features='@',
        absorbing_features='$',
        wall_features='#ABCDEFGHIJK',
        discount_rate=1.0,
        step_cost=-1,
        wall_bump_cost=-10
    )
    def __init__(self, participant_data : ParticipantDataBase):
        self.participant_data = participant_data
    def coarse_fine_trajectory_prob_matrix(
        self,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice
    ):
        ntrials = len(self.participant_data.main_trials())
        coarse_fine_probs = np.zeros((ntrials, 2))
        for trial_i, trial in enumerate(self.participant_data.main_trials()):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ctm = ConstrualTrialModel(**{
                    **self.default_gw_params,
                    **trial.invtransformed_GridWorld_params
                })
                states = trial.invtransformed_state_traj[:-1]
                actions = trial.invtransformed_action_traj
                coarse_prob = ctm.trajectory_prob(
                    states, actions,
                    construal_prior=coarse_identifier,
                    construal_cost_weight=construal_cost_weight,
                    construal_inverse_temp=construal_inverse_temp,
                    action_inverse_temp=action_inverse_temp,
                    action_random_choice=action_random_choice
                )
                fine_prob = ctm.trajectory_prob(
                    states, actions,
                    construal_prior=fine_identifier,
                    construal_cost_weight=construal_cost_weight,
                    construal_inverse_temp=construal_inverse_temp,
                    action_inverse_temp=action_inverse_temp,
                    action_random_choice=action_random_choice
                )
                coarse_fine_probs[trial_i, 0] = coarse_prob
                coarse_fine_probs[trial_i, 1] = fine_prob
        return coarse_fine_probs
    def trials_construal_set_values(
        self,
        construal_cost_weight,
        construal_inverse_temp,
    ):
        ntrials = len(self.participant_data.main_trials())
        coarse_fine_values = np.zeros((ntrials, 2))
        for trial_i, trial in enumerate(self.participant_data.main_trials()):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ctm = ConstrualTrialModel(**{
                    **self.default_gw_params,
                    **trial.invtransformed_GridWorld_params
                })
                coarse_val = ctm.construal_set_value(
                    construal_prior=coarse_identifier,
                    construal_cost_weight=construal_cost_weight,
                    construal_inverse_temp=construal_inverse_temp,
                )
                fine_val = ctm.construal_set_value(
                    construal_prior=fine_identifier,
                    construal_cost_weight=construal_cost_weight,
                    construal_inverse_temp=construal_inverse_temp,
                )
                coarse_fine_values[trial_i, 0] = coarse_val
                coarse_fine_values[trial_i, 1] = fine_val
        return coarse_fine_values
    
    def construal_trial_models(self):
        ctms = []
        for trial in self.participant_data.main_trials():
            ctm = ConstrualTrialModel(**{
                **self.default_gw_params,
                **trial.invtransformed_GridWorld_params
            })
            ctms.append(ctm)
        return ctms


    def trials_set_nextset_probability(
        self,
        construal_set_stickiness,
        construal_cost_weight,
        construal_inverse_temp,
    ):
        # P(C' | C)
        trials_set_values = self.trials_construal_set_values(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
        )
        set_nextset_switch_cost = (1 - np.eye(2))*construal_set_stickiness
        trials_set_nextset_values = \
            trials_set_values[:, None, :] - set_nextset_switch_cost[None, :, :]
        trials_set_nextset_probs = np.exp(trials_set_nextset_values)
        trials_set_nextset_probs /= trials_set_nextset_probs.sum(-1, keepdims=True)
        assert not np.isnan(trials_set_nextset_probs).any()
        return trials_set_nextset_probs
    
    def trials_set_nextset_traj_probability(
        self,
        construal_set_stickiness,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice
    ):
        # P(C' | C)
        trials_set_nextset_probs = self.trials_set_nextset_probability(
            construal_set_stickiness=construal_set_stickiness,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
        )

        # P(t | C)
        trials_set_traj_probs = self.coarse_fine_trajectory_prob_matrix(
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice
        )

        # P(C', t | C)
        trials_set_nextset_traj_probs = np.einsum(
            "tcn,tn->tcn",
            trials_set_nextset_probs,
            trials_set_traj_probs
        )
        return trials_set_nextset_traj_probs
        
    def trials_log_probability(
        self,
        construal_set_stickiness,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice,
        initial_set_prob=(.5, .5)
    ):
        trials_set_nextset_traj_probs = self.trials_set_nextset_traj_probability(
            construal_set_stickiness=construal_set_stickiness,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice
        )

        # probabilities get very small but need to be marginalized,
        # so we extract constants at each timestep to be added to 
        # the log-probability later
        norm = trials_set_nextset_traj_probs.sum(-1)
        constants = norm.max(-1)
        trials_set_nextset_traj_probs_sc = trials_set_nextset_traj_probs*(1/constants[:, None, None])
        ntrials, nsets, _ = trials_set_nextset_traj_probs_sc.shape
        marg_prob = np.ones(nsets)
        for t in range(ntrials - 1, -1, -1):
            marg_prob = trials_set_nextset_traj_probs_sc[t]@marg_prob
        marg_prob = marg_prob@np.array(initial_set_prob)
        
        assert marg_prob > 0
        marg_logprob = np.log(marg_prob) + np.log(constants).sum()
        return marg_logprob
    @method_cache
    def construal_set_hmm(
        self, 
        construal_set_stickiness,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice,
        initial_set_prob=(.5, .5)
    ):
        hmm = ParticipantSetHiddenMarkovModel(
            participant_model=self,
            model_parameters=dict(
                construal_set_stickiness=construal_set_stickiness,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
                action_inverse_temp=action_inverse_temp,
                action_random_choice=action_random_choice,
            ),
            initial_sets=initial_set_prob
        )
        return hmm
    def trials_model_stats(
        self,
        construal_set_stickiness,
        construal_cost_weight,
        construal_inverse_temp,
        action_inverse_temp,
        action_random_choice,
        initial_set_prob=(.5, .5)
    ):
        hmm = self.construal_set_hmm(
            construal_set_stickiness,
            construal_cost_weight,
            construal_inverse_temp,
            action_inverse_temp,
            action_random_choice,
            initial_set_prob=initial_set_prob
        )
        traj_margs = hmm.trajectory_marginals(1)
        stats_rec = []
        ptrials = self.participant_data.main_trials()
        for transition, probs in traj_margs.items():
            trial_num = transition[1]
            trial : GridNavigationTrialDataBase = ptrials[trial_num]
            stats = {
                (last_cset, cset[0]): probs[last_cset, cset]
                for last_cset, cset in probs.table_index.product()
            }
            prob_coarse = stats[('coarse', 'coarse')] + stats[('fine', 'coarse')]

            # calculate exp cog cost for each cset, then take the expected cost
            ctm = ConstrualTrialModel(**{
                **self.default_gw_params,
                **trial.invtransformed_GridWorld_params
            })
            coarse_cost = ctm.construal_set_cognitive_cost(
                construal_prior=coarse_identifier,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
            )
            fine_cost = ctm.construal_set_cognitive_cost(
                construal_prior=fine_identifier,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
            )
            expected_cognitive_cost = coarse_cost*prob_coarse + fine_cost*(1 - prob_coarse)

            coarse_value = ctm.construal_set_value(
                construal_prior=coarse_identifier,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
            )
            fine_value = ctm.construal_set_value(
                construal_prior=fine_identifier,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
            )

            prob_switch = stats[('coarse', 'fine')] + stats[('fine', 'coarse')]
            expected_switch_cost = prob_switch*construal_set_stickiness
            stats_rec.append({
                'sessionId': self.participant_data.sessionId,
                'condition_name': self.participant_data.condition_name,
                'trial': trial_num,
                'grid_name': trial.grid_name,
                'expected_cognitive_cost': expected_cognitive_cost,
                'expected_coarse_cost': coarse_cost,
                'expected_fine_cost': fine_cost,
                'expected_switch_cost': expected_switch_cost,
                'expected_coarse_value': coarse_value,
                'expected_fine_value': fine_value,
                'prob_coarse': prob_coarse,
                'prob_fine': 1 - prob_coarse,
                'prob_switch_to_coarse': stats[('fine', 'coarse')],
                'prob_switch_to_fine': stats[('coarse', 'fine')],
                'prob_switch': prob_switch,
                'initial_rt_milliseconds': trial.initial_rt_milliseconds,
                'max_noninit_rt_milliseconds': trial.max_noninit_rt_milliseconds,
                # **stats,
            })
        return stats_rec

class TimeInhomogeneousMarkovChain(Table):
    @classmethod
    def from_state_list(
        cls,
        state_list : list,
        data : np.ndarray,
    ):
        assert data.shape[1:] == (len(state_list), len(state_list))
        return cls(
            data=data,
            table_index=TableIndex(
                field_names=["timestep", "state", "next_state"],
                field_domains=[
                    domaintuple(range(data.shape[0])),
                    domaintuple(state_list),
                    domaintuple(state_list)
                ]
            )
        )
    
class MarkovChainStateLikelihoods(Table):
    @classmethod
    def from_state_list(
        cls,
        state_list : list,
        data : np.ndarray,
    ):
        assert data.shape[1:] == (len(state_list), )
        return cls(
            data=data,
            table_index=TableIndex(
                field_names=["timestep", "state"],
                field_domains=[
                    domaintuple(range(data.shape[0])),
                    domaintuple(state_list),
                ]
            )
        )
    def _repr_html_(self):
        df_kws = dict(float_format=lambda f: f"{f:.2g}")
        return super()._repr_html_(df_kws=df_kws)
    
class ParticipantSetHiddenMarkovModel:
    def __init__(
        self,
        participant_model : ParticipantModel,
        model_parameters : dict,
        initial_sets = (.5, .5)
    ):
        self.participant_model = participant_model
        self.model_parameters = model_parameters
        self.initial_sets = np.array(initial_sets)
        assert np.isclose(self.initial_sets.sum(), 1.0)
    def __len__(self):
        return self.set_transition_matrix.shape[0]
    @cached_property
    def set_transition_matrix(self):
        set_transitions = self.participant_model.trials_set_nextset_probability(
            construal_cost_weight=self.model_parameters['construal_cost_weight'],
            construal_set_stickiness=self.model_parameters['construal_set_stickiness'],
            construal_inverse_temp=self.model_parameters['construal_inverse_temp'],
        )
        return TimeInhomogeneousMarkovChain.from_state_list(
            state_list=["coarse", "fine"],
            data=set_transitions
        )
    @cached_property
    def set_likelihood_matrix(self):
        set_likelihood = self.participant_model.coarse_fine_trajectory_prob_matrix(
            construal_cost_weight=self.model_parameters['construal_cost_weight'],
            construal_inverse_temp=self.model_parameters['construal_inverse_temp'],
            action_inverse_temp=self.model_parameters['action_inverse_temp'],
            action_random_choice=self.model_parameters['action_random_choice'],
        )
        return MarkovChainStateLikelihoods.from_state_list(
            state_list=["coarse", "fine"],
            data=set_likelihood
        )
    @method_cache
    def forward(self, timestep):
        if timestep == 0:
            last_msg = self.initial_sets
        else:
            last_msg = self.forward(timestep - 1)
        set_traj_likelihood = np.array(self.set_likelihood_matrix[timestep])
        lastset_set_prob = np.array(self.set_transition_matrix[timestep])
        cur_msg = set_traj_likelihood*(last_msg@lastset_set_prob)
        cur_msg = cur_msg/cur_msg.sum()
        cur_msg.setflags(write=False)
        return cur_msg
    @method_cache
    def backward_log(self, timestep):
        # we do the backwards pass in log space since things easily underflow
        if timestep < 0:
            timestep = self.set_likelihood_matrix.shape[0] + timestep
        if timestep == (self.set_likelihood_matrix.shape[0] - 1):
            return np.log(np.array([1, 1]))
        next_log_msg = self.backward_log(timestep + 1)
        set_traj_likelihood = np.array(self.set_likelihood_matrix[timestep + 1])
        lastset_set_prob = np.array(self.set_transition_matrix[timestep + 1])
        temp_log_msg = np.log(set_traj_likelihood)+next_log_msg
        log_const = np.max(temp_log_msg)
        cur_log_msg = np.log(lastset_set_prob@np.exp(temp_log_msg - log_const)) + log_const
        assert not np.isnan(cur_log_msg).any()
        cur_log_msg.setflags(write=False)
        return cur_log_msg
    def state_marginal(self, timestep):
        marg_logprob = np.log(self.forward(timestep))+self.backward_log(timestep)
        marg_logprob -= np.max(marg_logprob)
        marg_prob = np.exp(marg_logprob)
        assert marg_prob.sum() > 0, "Underflow"
        marg_prob = marg_prob/marg_prob.sum()
        return marg_prob
    def state_marginals(self):
        return Table(
            data=np.stack([self.state_marginal(i) for i in range(len(self))]),
            table_index=self.set_likelihood_matrix.table_index
        )
    def trajectory_marginal(self, start, end):
        assert end >= start >= 0, (start, end)
        # p(state_{start} | obs_{:start})
        forward_msg = np.log(self.forward(start))
        
        # p(obs_{end+1:} | state_{end})
        backward_msg = self.backward_log(end)
        
        # p(state_{start+1, end}, obs_{start+1, end} | state_{start})
        hidden_likelihood = np.array(self.set_transition_matrix)*np.array(self.set_likelihood_matrix)[:, :, None]
        nstates = self.set_transition_matrix.shape[1]
        transitions = end - start
        trajectory_loglikelihoods = np.zeros([nstates, ]*(transitions + 1))
        trajectory_loglikelihoods += forward_msg[(slice(None),) + (None,)*transitions]
        trajectory_loglikelihoods += backward_msg[(None,)*transitions + (slice(None),)]
        for i in range(transitions):
            set_traj_likelihood = np.array(self.set_likelihood_matrix[i + start + 1])
            lastset_set_prob = np.array(self.set_transition_matrix[i + start + 1])
            temp_prob = set_traj_likelihood[None, :] * lastset_set_prob
            temp_logprob = np.log(temp_prob)
            reshaper = [None,]*i + [slice(None), slice(None)] + [None,]*(transitions - i - 1)
            temp_logprob = temp_logprob[tuple(reshaper)]
            trajectory_loglikelihoods += temp_logprob
        trajectory_loglikelihoods = trajectory_loglikelihoods.reshape((2, -1))
        trajectory_loglikelihoods -= trajectory_loglikelihoods.max()
        
        trajectory_probs = np.exp(trajectory_loglikelihoods)
        assert (trajectory_probs.sum() > 0), "Underflow"
        trajectory_probs /= trajectory_probs.sum()
        
        return Table(
            data=trajectory_probs,
            table_index=TableIndex(
                field_names=["start_set", "trajectory"],
                field_domains=[
                    domaintuple(["coarse", "fine"]),
                    domaintuple(product(["coarse", "fine"], repeat=transitions))
                ]
            )
        )
    def trajectory_marginals(self, slice_size=1):
        traj_margs = []
        for i in range(0, len(self) - slice_size):
            traj_marg = self.trajectory_marginal(i, i + slice_size)
            traj_margs.append(((i, i + slice_size), traj_marg))
        assert len(set([tm[1].table_index.fields for tm in traj_margs])) == 1, "Marginal tables are different"
        subtable_index = traj_margs[0][1].table_index.fields
        return Table(
            data=np.stack([np.array(tm[1]) for tm in traj_margs]),
            table_index=TableIndex(
                fields=[
                    Field("transition", [tm[0] for tm in traj_margs]),
                    *subtable_index
                ]
            )
        )