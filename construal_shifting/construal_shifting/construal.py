"""Maze construal functions

"""
import joblib
import tqdm 
import functools
from frozendict import frozendict
import matplotlib.pyplot as plt

from msdm.algorithms import ValueIteration, PolicyIteration, EntropyRegularizedPolicyIteration
from msdm.core.distributions import SoftmaxDistribution, DictDistribution
from msdm.core.utils.funcutils import cached_property
# from msdm.domains import GridWorld
from construal_shifting.fastgridworld import GridWorld2 as GridWorld
import copy
from construal_shifting import utils

joblibmemory = joblib.Memory('.', verbose=0)

@functools.lru_cache(maxsize=int(1e7))
def maze_construals(tilestring, max_size=1e25, ignore='.@$#sg'):
    """
    Enumerates tilestring construals of a broken-block 
    maze up to a maximum number of distinct obstacles
    """
    # we cache the output of the generator
    return tuple(_maze_construals(tilestring, max_size=1e25, ignore='.@$#sg'))

def _maze_construals(tilestring, max_size=1e25, ignore='.@$#sg'):
    loc_f = utils.segment_tilestring(tilestring)
    obs_tokens = set([f for f in loc_f.values() if f[0] not in ignore])
    blocks = {c[0].upper() for c in obs_tokens}
    def block_fmap(b):
        yield lambda f: '.' if f[0].upper() == b[0].upper() else f #ignore block b entirely
        # yield blocks with different numbers of lower case pieces
        pieces = [t for t in obs_tokens if t[0] == b.lower()]
        for pp in utils.powerset(pieces, decreasing_size=False):
            def make_fmap(pp):
                def fmap(f):
                    if f in pp:
                        return f
                    elif f[0].upper() == b[0].upper():
                        return f[0].upper()
                    else:
                        return f
                return fmap
            yield make_fmap(pp)
    def in_max_size(*fmaps):
        d = copy.copy(loc_f)
        for fmap in fmaps:
            d = {xy: fmap(f) for xy, f in d.items()}
        return len(set(d.values()) - set(list(ignore))) <= max_size
    for fmaps in utils.bounded_product(*(block_fmap(b) for b in blocks), in_bound=in_max_size):
        d = copy.copy(loc_f)
        for fmap in fmaps:
            d = {xy: fmap(f) for xy, f in d.items()}
        yield utils.dict_to_tilestring(d)
        
@functools.lru_cache(maxsize=int(1e7))
def construal_size(c, ignore='.@$#sg'):
    d = utils.segment_tilestring(c)
    return len(set([x for x in d.values() if x[0] not in ignore]))

def construal_utility(construal_tilestring, evaluation_tilestring, gw_params):
    """
    Calculates the value of a construal on the evaluation tilestring.
    """
    gw_params = frozendict(gw_params)
    return _cached_construal_utility(construal_tilestring, evaluation_tilestring, gw_params)

def _cached_construal_utility(construal_tilestring, evaluation_tilestring, gw_params):
    gw = GridWorld(
        [r.strip() for r in evaluation_tilestring.split('\n') if len(r.strip()) > 0],
        **gw_params
    )
    c_gw = GridWorld(
        [r.strip() for r in construal_tilestring.split('\n') if len(r.strip()) > 0],
        **gw_params
    )
    pi_res = PolicyIteration(
        max_iterations=200
    ).plan_on(c_gw)
    v0 = pi_res.policy.evaluate_on(gw).initial_value
    return v0

#create a disk and memory cache
joblib_func = joblibmemory.cache()(_cached_construal_utility)
_cached_construal_utility = functools.wraps(_cached_construal_utility)(joblib_func)
_cached_construal_utility = functools.lru_cache(maxsize=int(1e7))(_cached_construal_utility)

class ConstrualDistribution(SoftmaxDistribution):
    def __init__(
        self,
        construal_values,
        evaluation_tilestring,
        gw_params,
        construal_size_weight
    ):
        """Softmax over construals with helper methods
        """
        super(ConstrualDistribution, self).__init__(construal_values)
        self.evaluation_tilestring = evaluation_tilestring
        self.gw_params = gw_params
        self.construal_size_weight = construal_size_weight
    
    def plot_obstacles(self, plotter=None, ax=None, figsize=(7, 7)):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if plotter is None:
            gw = GridWorld(
                tile_array=[r.strip() for r in self.evaluation_tilestring.split('\n') if len(r.strip()) > 0],
                **self.gw_params
            )
            plotter = gw.plot(ax=ax, featurecolors={})
        return plotter.plot_state_map(self.obstacles())
    
    def obstacles(self):
        """
        Calculates the "expected construal" in terms of
        the by-location obstacles.
        """
        loc_seg = utils.tilestring_to_dict(self.evaluation_tilestring)
        loc_obstacle_prob = {}
        for loc in loc_seg:
            if loc_seg[loc] == '.':
                continue
            loc_obstacle_prob[loc] = \
                self.marginalize(
                    lambda c: utils.tilestring_to_dict(c)[loc].isupper() or utils.tilestring_to_dict(c)[loc] == '#'
            ).get(True, 0.0)
        return loc_obstacle_prob
    
    @cached_property
    def behavioral_utility(self):
        """
        Expected behavioral utility on a given task.
        """
        exp_behavioral_utility = self.expectation(
            lambda c: construal_utility(c, self.evaluation_tilestring, self.gw_params)
        )
        return exp_behavioral_utility

    @cached_property
    def cognitive_cost(self):
        """
        Expected cognitive cost on a given task.
        """
        cc = self.expectation(
            lambda c: - self.construal_size_weight*construal_size(c)
        )
        return cc
    
    @cached_property
    def value(self):
        """
        Expected behavioral utility minus cognitive cost.
        """
        return self.behavioral_utility - self.cognitive_cost
@functools.lru_cache(maxsize=int(1e7))
def n_broken_pieces(c):
    obs_broken = set(t for t in utils.segment_tilestring(c).values() if t[0] in "abcdefghijk")
    return len(obs_broken)
def construal_level(c):
    if n_broken_pieces(c) > 0:
        return 'fine'
    return 'coarse'

def calculate_construal_dist(
    tilestring,
    gw_params,
    softmax_invtemp=1,
    construal_size_weight=1,
    finegrained_utility=0,
    coarsegrained_utility=0,
    print_progress=True
):
    """
    Calculates construal distribution according to:
    $$
    \pi(c \mid s) = \sum_{l} \pi(c, l \mid s)
    $$

    where $c$ is a construal,
    $s$ is a state including the agent and all obstacles, 
    $l \in \{\text{fine}, \text{coarse}\}$ is the construal-level,
    and:

    $$
    \pi(c, l \mid s) \propto p(c \mid l)\exp\{\beta[U(\pi_c) - \beta_C |c| + U(l)] \}
    $$

    where $p(c \mid l)$ is 1 if the construal and the construal-level
    match and 0 otherwise, $\pi(l)$ is the construal-level policy.
    $\beta$ is the global inverse temperature, while $\beta$ is the
    construal-size weight.
    

    Parameters
    ----------
    tilestring : str 
        A multi-line string representing the current state of
        the gridworld, $s$.
    gw_params : dict
        Parameters to be passed to GridWorld
    softmax_invtemp : float
        Softmax inverse temperature, $\beta$.
    construal_size_weight : float
        The weight on the construal size, $\beta_C$
    finegrained_utility : float
        The utility associated with using a finegrained construal,
        $U(l = \text{fine})$
    coarsegrained_utility : float
        The utility associated with using a coarsegrained construal,
        $U(l = \text{coarse})$
    print_progress : bool
        Whether to print progress on computing construal values

    Returns
    -------
    Distribution[str, float]
        A distribution over construals, $c$, represented as multi-line strings
    """
    c_iterator = maze_construals(tilestring)
    if print_progress:
        c_iterator = tqdm.tqdm(list(c_iterator))
    vor = {}
    for c in c_iterator:
        if construal_level(c) == 'fine':
            vor[c] = \
                construal_utility(c, tilestring, gw_params) - \
                construal_size_weight*construal_size(c) + \
                finegrained_utility
        else:
            vor[c] = \
                construal_utility(c, tilestring, gw_params) - \
                construal_size_weight*construal_size(c) + \
                coarsegrained_utility
    construal_dist = ConstrualDistribution(
        {c: softmax_invtemp*v for c, v in vor.items()},
        evaluation_tilestring=tilestring,
        gw_params=gw_params,
        construal_size_weight=construal_size_weight
    )
    return construal_dist

