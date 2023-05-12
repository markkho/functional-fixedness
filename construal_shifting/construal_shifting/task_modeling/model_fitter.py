import random
import numpy as np
import dataclasses
from scipy.optimize import fmin_l_bfgs_b

from frozendict import frozendict
from tqdm.notebook import tqdm
from construal_shifting.task_modeling.participant_model import ParticipantModel
from construal_shifting.task_modeling.construal_trial_model import ConstrualTrialModel
from construal_shifting.task_modeling.base_dataclasses import ParticipantDataBase
from typing import Sequence
from msdm.core.utils.funcutils import method_cache

class ModelFitter:
    def __init__(
        self,
        all_participant_data : Sequence[ParticipantDataBase]
    ) -> None:
        self.all_participant_data = all_participant_data
    
    @method_cache
    def evaluate_model_params(self, model_params):
        total_log_prob = 0
        for participant_data in self.all_participant_data:
            pmod = ParticipantModel(participant_data)
            total_log_prob += pmod.trials_log_probability(**model_params)
        return total_log_prob

    def fit_params_once(
        self,
        fixed_params,
        initial_params,
        param_bounds,
        maxfun,
    ) -> "FitResult":
        param_names = sorted(initial_params.keys())
        iterator = tqdm(total=maxfun)
        def func(x):
            iterator.update(n=1)
            model_params = dict(zip(param_names, x))
            model_params = {**fixed_params, **model_params}
            log_prob = self.evaluate_model_params(frozendict(model_params))
            iterator.set_description(f"NLL: {-log_prob:.0f}; {param_dict_to_str(model_params)}")
            return -log_prob

        x0 = np.array([initial_params[n] for n in param_names])
        x, nll, info = fmin_l_bfgs_b(
            func=func,
            x0=x0,
            bounds=[param_bounds[n] for n in param_names],
            maxfun=maxfun,
            approx_grad=True
        )
        iterator.close()
        model_params = dict(zip(param_names, x))
        model_params = {**fixed_params, **model_params}
        return FitResult(
            model_params=model_params,
            neg_log_like=nll,
            x0=x0,
            info=info,
            initial_params=initial_params
        )

    def fit_params(
        self,
        fixed_params,
        params_to_fit,
        param_bounds,
        maxfun,
        runs,
        seed=None
    ) -> Sequence["FitResult"]:
        rng = random.Random(seed)
        results = []
        for _ in range(runs):
            initial_params = {}
            for name in params_to_fit:
                vmin, vmax = param_bounds[name]
                initial_params[name] = vmin + rng.random()*(vmax - vmin)
            res = self.fit_params_once(
                fixed_params={p: v for p, v in fixed_params.items() if p not in params_to_fit},
                initial_params=initial_params,
                param_bounds=param_bounds,
                maxfun=maxfun
            )
            results.append(res)
            ConstrualTrialModel.clear_instance_method_caches()
        return results

def param_dict_to_str(params : dict):
    pstr = []
    pnames = sorted(params.keys())
    for n in pnames:
        short_n = n.replace('construal', 'c').\
            replace('cost_weight', 'cost').\
            replace('inverse_temp', 'invt').\
            replace('random_choice', 'eps').\
            replace('action', 'a').\
            replace('set_stickiness', 'stick')
        pstr.append(f'{short_n}:{params[n]:.2f}')
    return ', '.join(pstr)

@dataclasses.dataclass
class FitResult:
    model_params : dict
    neg_log_like : float
    x0 : np.ndarray
    info : dict
    initial_params : dict