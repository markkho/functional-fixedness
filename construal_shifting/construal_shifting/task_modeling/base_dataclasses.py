from typing import Sequence
from abc import abstractmethod, abstractproperty
class GridNavigationTrialDataBase:
    @abstractproperty
    def invtransformed_state_traj(self): pass
    @abstractproperty
    def invtransformed_action_traj(self): pass
    @abstractproperty
    def invtransformed_GridWorld_params(self): pass
    @abstractproperty
    def initial_rt_milliseconds(self): pass
    @abstractproperty
    def max_noninit_rt_milliseconds(self): pass
    @abstractmethod
    def summary(self): pass

class ParticipantDataBase:
    sessionId : str
    condition_name : str
    @abstractmethod
    def main_trials(self) -> Sequence[GridNavigationTrialDataBase]: pass
    @abstractmethod
    def summary(self): pass