import urllib.request
import csv
import math
import json
import copy
import sys
import functools
import configparser
import datetime
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union, Callable, Sequence, Mapping, Set, Type, NamedTuple

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from construal_shifting.fastgridworld import GridWorld2 as GridWorld
from construal_shifting import gridutils, sampsat
from construal_shifting.task_modeling.base_dataclasses import ParticipantDataBase, GridNavigationTrialDataBase
from msdm.algorithms import PolicyIteration
from msdm.core.utils.funcutils import cached_property, method_cache

def download_data(
    trialdata_file, 
    EXPERIMENT_CODE_VERSION=None
):
    EXPERIMENT_URL = "myexperiment.com"
    CONFIG_FILE = "config.json"
    PSITURK_CONFIG_FILE = "../../psiturkapp/config.txt"

    # Load credentials and experiment code version
    if EXPERIMENT_CODE_VERSION is None:
        exp_config = json.load(open(CONFIG_FILE, 'r'))
        EXPERIMENT_CODE_VERSION = exp_config['params']['EXPERIMENT_CODE_VERSION']
    psiturk_config = configparser.ConfigParser()
    assert psiturk_config.read(PSITURK_CONFIG_FILE)
    list(psiturk_config.keys())
    login_username = psiturk_config["Server Parameters"]["login_username"]
    login_pw = psiturk_config["Server Parameters"]["login_pw"]
    print("Credentials:", login_username, "*"*len(login_pw[-5:]) + login_pw[-5:], sep="\n  ")

    # Connect to database and pull data into csv files
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, EXPERIMENT_URL, login_username, login_pw)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    data_url = f"{EXPERIMENT_URL}data/{EXPERIMENT_CODE_VERSION}/trialdata"
    print(f"Loading data from {data_url}")
    opener = urllib.request.build_opener(handler)
    opener.open(data_url)
    urllib.request.install_opener(opener)
    res = urllib.request.urlretrieve(
        url=data_url,
        filename=trialdata_file
    )
    return res

def download_condition_counts(
    condition_count_file, 
    EXPERIMENT_CODE_VERSION=None
):
    EXPERIMENT_URL = "myexperiment.com"
    CONFIG_FILE = "config.json"
    PSITURK_CONFIG_FILE = "../../psiturkapp/config.txt"

    # Load credentials and experiment code version
    if EXPERIMENT_CODE_VERSION is None:
        exp_config = json.load(open(CONFIG_FILE, 'r'))
        EXPERIMENT_CODE_VERSION = exp_config['params']['EXPERIMENT_CODE_VERSION']
    psiturk_config = configparser.ConfigParser()
    assert psiturk_config.read(PSITURK_CONFIG_FILE)
    list(psiturk_config.keys())
    login_username = psiturk_config["Server Parameters"]["login_username"]
    login_pw = psiturk_config["Server Parameters"]["login_pw"]
    print("Credentials:", login_username, "*"*len(login_pw[-5:]) + login_pw[-5:], sep="\n  ")

    # Connect to database and pull data into csv files
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, EXPERIMENT_URL, login_username, login_pw)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

    data_url = f"{EXPERIMENT_URL}data/{EXPERIMENT_CODE_VERSION}/conditiondata"
    print(f"Loading data from {data_url}")
    opener = urllib.request.build_opener(handler)
    opener.open(data_url)
    urllib.request.install_opener(opener)
    res = urllib.request.urlretrieve(
        url=data_url,
        filename=condition_count_file
    )
    return res

def calc_condition_counts(
    CONFIG_NAME,
    condition_count_file
):
    config = json.load(open(CONFIG_NAME))
    def get_condition_name(tl):
        for tt in tl[:-1]:
            if tt.get('type') == 'SaveGlobalStore':
                return tt['condition_name']
    condition_to_name = {condition: get_condition_name(tl) for condition, tl in enumerate(config['timelines'])}
    conds = [line for line in csv.reader(open(condition_count_file, 'r'))]
    cond_counts = Counter([int(c[1]) for c in conds])
    min_count = min(cond_counts.values())
    minima_names = Counter([condition_to_name[cond] for cond, count in cond_counts.items() if count == min_count])
    cond_name_counts = Counter([condition_to_name[int(c[1])] for c in conds])
    return dict(
        cond_name_counts=cond_name_counts,
        minima_names=minima_names
    )

def to_plt_rgb(rgba_string):
    if "rgba" not in rgba_string:
        return rgba_string
    rgba_string = rgba_string.replace("rgba(", "")
    rgba_string = rgba_string.replace(")", "")
    r, g, b, a= [float(i) for i in rgba_string.split(",")]
    rgba = (r/256, g/256, b/256, a)
    return rgba

def plot_trial(data, ax=None, trajectory_params=None):
    taskparams = data['data']['taskparams']
    gw_params = {p: taskparams[p] for p in ["tile_array", "absorbing_features", "initial_features"]}
    gw = GridWorld(**gw_params)
    gwp = gw.plot(featurecolors={f: to_plt_rgb(c) for f, c in taskparams['feature_colors'].items()}, ax=ax)
    state_traj = [t['state'] for t in data['data']['navigationData']] + [data['data']['navigationData'][-1]['nextstate'], ]
    gwp.plot_trajectory(
        state_traj,
        **dict(
            lw=2,
            color='green',
            **(trajectory_params if trajectory_params else {})
        )
    )
    # plot bumps
    for t in data['data']['navigationData']:
        if t['state'] == t['nextstate']:
            gwp.annotate(t['state'], text="o", color='green')
    return gwp

@functools.lru_cache(maxsize=int(1e5))
def calculate_critical_notch_locs(tile_array):
    gw_params = dict(
        tile_array=tile_array,
        wall_features="ABCDEFG#",
        absorbing_features="$",
        initial_features="@",
        step_cost=-1,
    )
    gw_disc = GridWorld(
        **gw_params,
        discount_rate=1 - 1e-8
    )
    pi = PolicyIteration().plan_on(gw_disc)
    gw = GridWorld(
        **gw_params,
        discount_rate=1
    )
    policy_eval = pi.policy.evaluate_on(gw)
    try:
        occupancy = policy_eval.occupancy
    except AttributeError:
        occupancy = policy_eval.state_occupancy
    unavoidable_locs = {loc for loc, prob in occupancy.items() if prob == 1.0}
    notch_locs = set([])
    for f in 'abcdefg':
        if f in gw.feature_locations:
            notch_locs.update(gw.feature_locations[f])
    crit_notch_locs = notch_locs & unavoidable_locs
    crit_notch_locs = {(l['x'], l['y']) for l in crit_notch_locs}
    return crit_notch_locs

class GridNavigationTrialData(GridNavigationTrialDataBase):
    def __init__(self, data):
        """
        Structure of GridNavigationTrial data:
        {  // data, internal_node_id, time_elapsed, trial_index, trial_type
            data: {  // datatype, navigationData, sessionId, taskparams, trialparams
                datatype: str,
                navigationData: [
                    {  
                        action: [int],
                        nextstate: [int],
                        nextstate_type: str,
                        response_datetime: int,
                        reward: int,
                        sessionId: str,
                        start_datetime: int,
                        state: [int],
                        state_type: str,
                        trialnum: int,
                    }
                ],
                sessionId: str,
                taskparams: {
                    absorbing_features: [str],
                    default_features: [str],
                    feature_colors: {str: str},
                    feature_rewards: {str: int},
                    initial_features: [str],
                    step_cost: int,
                    tile_array: [str],
                    wall_bump_cost: int,
                    wall_features: [str],
                },
                trialparams: {
                    GOALSIZE: float,
                    GOAL_COUNTDOWN_SEC: int,
                    INITIALGOAL_COUNTDOWN_MILLISEC: int,
                    INITIALGOAL_COUNTDOWN_SEC: int,
                    MAX_TIMESTEPS: int,
                    OBJECT_ANIMATION_TIME: int,
                    TILE_SIZE: int,
                    dollarsPerPoint: float,
                    goalCountdown: bool,
                    grid_name: str,
                    hideObstaclesOnMove: bool,
                    initialPoints: int,
                    navigationText: str,
                    participantStarts: bool,
                    round: int,
                    roundtype: str,
                    showPoints: bool,
                    swap_start_goal: bool,
                    transform: str,
                },
            },
            internal_node_id: str,
            time_elapsed: int,
            trial_index: int,
            trial_type: str,
        }
        """
        self.data = data
        self.navigationData = self.data['data']['navigationData']
        assert len(self.navigationData) == self.navigationData[-1]['trialnum'] + 1, "Missing trials"
        assert [t['trialnum'] for t in self.navigationData] == list(range(len(self.navigationData)))
        
    @property
    def total_reward(self):
        return sum([st['reward'] for st in self.navigationData])
    @property
    def timesteps(self):
        return len(self.navigationData)
    @property
    def total_time_milliseconds(self):
        return self.navigationData[-1]['response_datetime'] - self.navigationData[0]['start_datetime']
    @property
    def initial_rt_milliseconds(self):
        return self.navigationData[0]['response_datetime'] - self.navigationData[0]["start_datetime"]
    @property
    def visited_critical_notch(self):
        visited_locs = set([tuple(step['state']) for step in self.navigation_data])
        tile_array = self.data['data']['taskparams']['tile_array']

        critical_notch_locs = calculate_critical_notch_locs(tuple(tile_array))
        return len(visited_locs & critical_notch_locs) > 0
    @property
    def notch_occupancy(self):
        return sum([step['state_type'] in 'abcdef' for step in self.navigation_data])
    @property
    def navigation_data(self):
        return copy.deepcopy(self.data['data']['navigationData'])
    @cached_property
    def invtransformed_state_traj(self):
        transform_loc = getattr(gridutils.transform_location, self.transform)
        tile_array = self.data['data']['taskparams']['tile_array']
        state_traj = [tuple(t['state']) for t in self.navigationData] + [self.navigationData[-1]['nextstate'], ]
        state_traj = [transform_loc.inv(tile_array, s) for s in state_traj]
        state_traj = [(int(x), int(y)) for x, y in state_traj]
        return state_traj
    @cached_property
    def invtransformed_action_traj(self):
        transform_dir = getattr(gridutils.transform_direction, self.transform)
        action_traj = [tuple(t['action']) for t in self.navigationData]
        action_traj = [transform_dir.inv(a) for a in action_traj]
        action_traj = [(int(dx), int(dy)) for dx, dy in action_traj]
        return action_traj
    @property
    def GridWorld_params(self):
        taskparams = self.data['data']['taskparams']
        gw_params = {p: tuple(copy.deepcopy(taskparams[p])) for p in ["tile_array", "absorbing_features", "initial_features"]}
        return gw_params
    @property
    def plot_params(self):
        taskparams = self.data['data']['taskparams']
        return {
            'featurecolors': {f: to_plt_rgb(c) for f, c in taskparams['feature_colors'].items()}
        }
    @cached_property
    def invtransformed_GridWorld_params(self):
        taskparams = self.data['data']['taskparams']
        gw_params = {p: tuple(copy.deepcopy(taskparams[p])) for p in ["tile_array", "absorbing_features", "initial_features"]}
        transform_grid = getattr(gridutils.transform_grid, self.transform)
        gw_params['tile_array'] = tuple(transform_grid.inv(gw_params['tile_array']))
        return gw_params
    @property
    def trialparams(self):
        return self.data['data']['trialparams']
    @property
    def trial_index(self):
        return self.data['trial_index']
    @property
    def grid_name(self):
        return self.trialparams['grid_name']
    @property
    def swap_start_goal(self):
        return self.trialparams['swap_start_goal']
    @property
    def transform(self):
        return self.trialparams['transform']
    @property
    def max_noninit_rt_milliseconds(self):
        return max([t['response_datetime'] - t['start_datetime'] for t in self.navigationData[1:]])
    def summary(self):
        return dict(
            trial_index=self.trial_index,
            grid_name=self.grid_name,
            total_reward=self.total_reward,
            timesteps=self.timesteps,
            total_time_milliseconds=self.total_time_milliseconds,
            initial_rt_milliseconds=self.initial_rt_milliseconds,
            max_noninit_rt_milliseconds=self.max_noninit_rt_milliseconds,
            notch_occupancy=self.notch_occupancy,
            visited_critical_notch=self.visited_critical_notch,
            swap_start_goal=self.swap_start_goal,
            transform=self.transform
        )
    def plot_trial(self, ax=None, trajectory_params=None):
        taskparams = self.data['data']['taskparams']
        gw_params = {p: taskparams[p] for p in ["tile_array", "absorbing_features", "initial_features"]}
        gw = GridWorld(**gw_params)
        gwp = gw.plot(featurecolors={f: to_plt_rgb(c) for f, c in taskparams['feature_colors'].items()}, ax=ax)
        state_traj = [t['state'] for t in self.navigationData] + [self.navigationData[-1]['nextstate'], ]
        gwp.plot_trajectory(
            state_traj,
            **dict(
                lw=2,
                color='green',
                **(trajectory_params if trajectory_params else {})
            )
        )
        # plot bumps
        for t in self.navigationData:
            if t['state'] == t['nextstate']:
                s = t['state']
                a = t['action']
                bump = [
                    s[0] + a[0]*.4+(.5-np.random.random()*.6 if a[0] == 0 else 0),
                    s[1] + a[1]*.4+(.5-np.random.random()*.6 if a[1] == 0 else 0)
                ]
                gwp.annotate(bump, text="o", color='green')
        
        # annotate figure
        tp = self.trialparams
        init_rt = self.initial_rt_milliseconds
        title = f"{self.data['trial_index']}, {tp['grid_name']}, {tp['transform']}, " + \
            f"{'swap' if tp['swap_start_goal'] else 'no-swap'}, {tp['roundtype']}\n" + \
            f"initial rt: {init_rt}, steps: {self.timesteps}, " +\
            f" total_time: {self.total_time_milliseconds}, reward: {self.total_reward}"
        gwp.title(title)
        return gwp
    
class MissingTrials(Exception):
    pass

class ParticipantData(ParticipantDataBase):
    def __init__(
        self,
        participant_id,
        trials
    ):
        trials = sorted(trials, key=lambda t: t['trial_index'])
        # check that all trials are present
        if [t['trial_index'] for t in trials] != list(range(len(trials))):
            raise MissingTrials(f"Trial Indices = {[t['trial_index'] for t in trials]}; expected {list(range(len(trials)))}")
        assert trials[-1]['trial_index'] == len(trials) - 1
        self.trials = trials
        self.prolific_id, self.assignment_id = participant_id.split(':')
        self.is_debug = "debug" in self.prolific_id
        self.completed = "SaveGlobalStore" in set([t['trial_type'] for t in self.trials])
        if not self.is_debug and self.completed:
            self.sessionId = self.final_global_store()['sessionId']
        else:
            self.sessionId = None
        
    @method_cache
    def main_trials(self) -> List[GridNavigationTrialData]:
        return self.training_trials() + self.test_trials()
    
    @method_cache
    def training_trials(self) -> List[GridNavigationTrialData]:
        tt = []
        for t in self.trials:
            if t['trial_type'] == "GridNavigation" and 'training' in t['data']['trialparams']['roundtype']:
                tt.append(GridNavigationTrialData(t))
        return tt
    
    @method_cache
    def test_trials(self) -> List[GridNavigationTrialData]:
        tt = []
        for t in self.trials:
            if t['trial_type'] == "GridNavigation" and t['data']['trialparams']['roundtype'] in ['test']:
                tt.append(GridNavigationTrialData(t))
        assert all([(ta.trial_index + 1) == tb.trial_index for ta, tb in zip(tt, tt[1:])])
        return tt
    
    @property
    def total_exp_time(self):
        tt = [t for t in self.trials if t['trial_type'] == 'fullscreen']
        assert len(tt) == 2
        return datetime.timedelta(milliseconds=(tt[1]['time_elapsed'] - tt[0]['time_elapsed']))
    
    def question_responses(self):
        tt = [t for t in self.trials if t['trial_type'] == 'CustomSurvey']
        qresp = {}
        for t in tt:
            assert len(set(qresp.keys()) & set(t['data'].keys())) == 0
            qresp.update(t['data'])
        return qresp

    def final_global_store(self):
        tt = [t for t in self.trials if t['trial_type'] == 'SaveGlobalStore']
        assert len(tt) == 1
        return tt[0]['data']

    @property
    def condition_name(self) -> str:
        return self.final_global_store()['condition_name']
    def summary(self) -> dict:
        gstore = {
            k: v for k, v in self.final_global_store().items()
            if k in ["sessionId","bonusDollars","condition","condition_name"]
        }
        test = self.test_trials()
        training = self.training_trials()
        
        return {
            **gstore,
            **self.question_responses(),
            "prolific_id": self.prolific_id,
            # "assignment_id": self.assignment_id,
            "n_training": len(training),
            "n_test": len(test),
            "mean_test_steps": np.mean([t.timesteps for t in test]),
            "total_exp_time": self.total_exp_time
        }
    def test_trials_summary(self) -> List[dict]:
        tts = []
        psummary = self.summary()
        for t in self.test_trials():
            ts = t.summary()
            ts.update(dict(
                sessionId=psummary['sessionId'],
                training_condition=psummary["training_condition"],
                condition=psummary("condition"),
            ))
            tts.append(ts)
        return tts
    def plot_trials(self, nrows=2, figsize_mult=5):
        main_trials = list(self.main_trials())
        ncols = math.ceil(len(main_trials)/nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figsize_mult, nrows*figsize_mult))
        axes = list(axes.flatten())
        for t in main_trials:
            ax = axes.pop(0)
            gwp = t.plot_trial(ax=ax)
        suptitle_kw = [
            'sessionId', 'bonusDollars', 'condition', 
            'generalComments', 'greenSquareThink',
            'winPointsThink', 'brokenBlocksThink',
            'gender', "age", 'mean_test_steps', 'total_exp_time'
        ]
        summary = self.summary()
        suptitle = '\n'.join([f"{k}: {summary[k]}" if not isinstance(summary[k], float) else f"{k}: {summary[k]:.2f}" for k in suptitle_kw])
        plt.suptitle(suptitle)
        plt.tight_layout()

class ExperimentDataLoader:
    def __init__(
        self,
        trialdata_file
    ):
        csv.field_size_limit(sys.maxsize)
        trialdata = [line for line in csv.reader(open(trialdata_file, 'r'))]
        trials_by_participant = defaultdict(list)
        for participant_id, page_id, timestamp, data in trialdata:
            trials_by_participant[participant_id].append(json.loads(data))
        participant_data = []
        for participant_id, trials in trials_by_participant.items():
            try:
                participant_data.append(ParticipantData(
                    participant_id=participant_id, 
                    trials=trials
                ))
            except MissingTrials as e:
                print(f'Participant {participant_id} missing trials')
                print(e)
        self.participant_data = participant_data
    
    def participants(self) -> Iterable[ParticipantData]:
        return self.participant_data
    
    def completed_participant_data(self) -> Iterable[ParticipantData]:
        for p in self.participant_data:
            if p.is_debug:
                continue
            if not p.completed:
                continue
            yield p
    
    def test_trials(self) -> pd.DataFrame:
        p_test = []
        for p in self.completed_participant_data():
            df = pd.DataFrame([t.summary() for t in p.test_trials()])
            df['sessionId'] = p.sessionId
            df['test_trial_num'] = range(len(df))
            p_test.append(df)
        return pd.concat(p_test).reset_index(drop=True)
    def plot_participant_trials(self):
        for p in self.completed_participant_data():
            p.plot_trials()
    def summary_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(p.summary() for p in self.completed_participant_data())
    def bonus_string(self) -> str:
        bonusdf = self.summary_dataframe()[["prolific_id", "bonusDollars"]]
        bonusdf = bonusdf[bonusdf.bonusDollars > 0]
        assert all(bonusdf.bonusDollars < 2.5)
        return '\n'.join([f"{pid},{bonus:.2f}" for _, pid, bonus in bonusdf.to_records()])