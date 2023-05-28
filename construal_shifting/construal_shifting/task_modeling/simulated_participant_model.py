import dataclasses
import random
from msdm.core.utils.funcutils import cached_property, method_cache
from msdm.core.mdp.policy import SimulationResult
from construal_shifting.task_modeling.cached_gridworld import GridWorld
from typing import Sequence, Tuple
from construal_shifting.task_modeling.base_dataclasses import GridNavigationTrialDataBase, ParticipantDataBase
from construal_shifting.task_modeling.participant_model import coarse_identifier, fine_identifier
from construal_shifting.task_modeling.participant_model import ParticipantModel
from construal_shifting.task_modeling.construal_trial_model import ConstrualTrialModel 

class SimulatedParticipantData(ParticipantDataBase):
    @classmethod
    def from_real_participant(
        cls,
        participant_data : ParticipantDataBase,
        construal_cost_weight : float,
        construal_inverse_temp : float,
        action_inverse_temp : float,
        action_random_choice : float,
        construal_set_stickiness : float,
        seed : int
    ):
        rng = random.Random(seed)
        cset = rng.choice(['coarse', 'fine'])
        simulated_participant_trials = []
        for trial_index, trial in enumerate(participant_data.main_trials()):
            ctm = ConstrualTrialModel(
                **{
                    **ParticipantModel.default_gw_params,
                    **trial.invtransformed_GridWorld_params
                }
            )
            trial_sim = ctm.simulate_trial(
                last_cset=cset,
                construal_cost_weight=construal_cost_weight,
                construal_inverse_temp=construal_inverse_temp,
                action_inverse_temp=action_inverse_temp,
                action_random_choice=action_random_choice,
                construal_set_stickiness=construal_set_stickiness,
                trial_index=trial_index,
                rng=rng
            )
            cset = trial_sim.current_construal_set
            simulated_participant_trials.append(trial_sim)
        return SimulatedParticipantData(
            sessionId=participant_data.sessionId + f"_simulated_{seed}",
            condition_name=participant_data.condition_name,
            main_trials=simulated_participant_trials,
            construal_cost_weight=construal_cost_weight,
            construal_inverse_temp=construal_inverse_temp,
            action_inverse_temp=action_inverse_temp,
            action_random_choice=action_random_choice,
            construal_set_stickiness=construal_set_stickiness,
        )

    def __init__(
        self,
        sessionId : str,
        condition_name : str,
        main_trials : Sequence["SimulatedGridNavigationTrialData"],
        construal_cost_weight : float,
        construal_inverse_temp : float,
        action_inverse_temp : float,
        action_random_choice : float,
        construal_set_stickiness : float,
    ):
        self.sessionId = sessionId
        self.condition_name = condition_name
        self._main_trials = main_trials
        self.construal_cost_weight = construal_cost_weight
        self.construal_inverse_temp = construal_inverse_temp
        self.action_inverse_temp = action_inverse_temp
        self.action_random_choice = action_random_choice
        self.construal_set_stickiness = construal_set_stickiness

    def main_trials(self) -> Sequence["SimulatedGridNavigationTrialData"]:
        return self._main_trials
    def summary(self):
        return dict(
            sessionId=self.sessionId,
            condition_name=self.condition_name,
            construal_cost_weight=self.construal_cost_weight,
            construal_inverse_temp=self.construal_inverse_temp,
            action_inverse_temp=self.action_inverse_temp,
            action_random_choice=self.action_random_choice,
            construal_set_stickiness=self.construal_set_stickiness,
        )

@dataclasses.dataclass
class SimulatedGridNavigationTrialData(GridNavigationTrialDataBase):
    trial_params : dict
    coarse_set_cognitive_cost : float
    fine_set_cognitive_cost : float
    state_traj : Sequence[Tuple]
    action_traj : Sequence[Tuple]
    construal_cost_weight : float
    construal_inverse_temp : float
    action_inverse_temp : float
    action_random_choice : float
    construal_set_stickiness : float
    current_construal_set : str
    trial_index : int = None
    grid_name : str = None

    @cached_property
    def invtransformed_state_traj(self):
        return self.state_traj

    @cached_property
    def invtransformed_action_traj(self):
        return self.action_traj

    @cached_property
    def invtransformed_GridWorld_params(self):
        return self.trial_params

    @property
    def initial_rt_milliseconds(self):
        # the 'initial reaction time' is the expected construal cost
        if self.current_construal_set == 'coarse':
            rt = self.coarse_set_cognitive_cost
        elif self.current_construal_set == 'fine':
            rt = self.fine_set_cognitive_cost
        else:
            raise ValueError(self.current_construal_set)
        assert rt >= 0
        return rt

    @property
    def max_noninit_rt_milliseconds(self):
        return -1

    def summary(self):
        return dict(
            trial_index=self.trial_index,
            total_reward=sum(self.sim_result.reward),
            timesteps=len(self.sim_result),
            initial_rt_milliseconds=self.initial_rt_milliseconds,
            current_construal_set=self.current_construal_set,
            construal_cost_weight=self.construal_cost_weight,
            construal_inverse_temp=self.construal_inverse_temp,
            action_inverse_temp=self.action_inverse_temp,
            action_random_choice=self.action_random_choice,
            construal_set_stickiness=self.construal_set_stickiness,
        )