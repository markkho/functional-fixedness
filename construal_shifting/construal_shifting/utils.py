"""Utilities

"""
import math
import copy
from collections import defaultdict
from itertools import product, combinations
from functools import reduce, lru_cache
import matplotlib.pyplot as plt
import hashlib

# plotting
def axes_grid(
    size,
    ncols=None,
    nrows=None,
    figmult=8
):
    if ncols is None:
        ncols = math.ceil(size**.5)
    if nrows is None:
        nrows = size // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figmult, nrows*figmult))
    axes = axes.flatten()
    yield from axes

# General Utilities
def bounded_product(*seqs, in_bound=None):
    """
    Yields Cartesian product of n sequences, [(a0, b0, ..., n0), ...] as long as 
    in_bound(a, b, ..., n) evaluates to true. This assumes that the in_bound 
    function is monotonic in any single variable in seq when the other variables
    are fixed in value.
    """
    seqs = [list(seq) for seq in seqs] # potentially memory intensive
    if len(seqs) == 0:
        yield ()
        return
    if in_bound is None:
        in_bound = lambda *args: True
    def _bounded_product(*seqs, fixed):
        if len(seqs) == 1:
            for v in seqs[0]:
                if not in_bound(v, *fixed):
                    return
                yield (v, ) + fixed
            return
        for v in seqs[-1]:
            yield from _bounded_product(*seqs[:-1], fixed=((v, ) + fixed))
    yield from _bounded_product(*seqs, fixed=())
    
def powerset(s, decreasing_size=True):
    if decreasing_size:
        iterator = range(len(s), -1, -1)
    else:
        iterator = range(len(s)+1)
    for n in iterator:
        yield from combinations(s, r=n)

# Gridworld utilities
@lru_cache(maxsize=int(1e6))
def _cached_tilestring_to_dict(tilestring):
    tiles = [list(row.strip()) for row in tilestring.split('\n') if len(row.strip()) > 0]
    d = {}
    for y, row in enumerate(tiles):
        for x, f in enumerate(row):
            d[(x, len(tiles) - y - 1)] = f
    return d

def tilestring_to_dict(tilestring):
    # cached result is a simple, non-nested dictionary so this is ok
    return _cached_tilestring_to_dict(tilestring).copy()

def dict_to_tilestring(d):
    h = max(y for x,y in d.keys()) + 1
    w = max(x for x,y in d.keys()) + 1
    tiles = [['' for _ in range(w)] for _ in range(h)]
    for x, y in product(range(w), range(h)):
        tiles[h - y - 1][x] = d[(x, y)][0]
    return '\n'.join([''.join(r) for r in tiles])

@lru_cache(maxsize=int(1e6))
def _cached_segment_tilestring(tilestring):
    loc_f = tilestring_to_dict(tilestring)
    def get_adjacent(loc):
        return [(loc[0] + dx, loc[1] + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]

    segments = defaultdict(list)
    while loc_f:
        loc = next(iter(loc_f.keys()))
        f = loc_f.pop(loc)
        nlocs = get_adjacent(loc)
        contiguous = [loc]
        while nlocs:
            nloc = nlocs.pop()
            if (nloc in loc_f) and (loc_f[nloc] == f):
                loc_f.pop(nloc)
                contiguous.append(nloc)
                nlocs.extend(get_adjacent(nloc))
        segments[f].append(contiguous)
    locmap = {}
    for f, segs in segments.items():
        if len(segs) == 1:
            for loc in segs[0]:
                locmap[loc] = f
            continue
        for fi, seg in enumerate(segs):
            for loc in seg:
                locmap[loc] = f"{f}{fi}"
    return locmap

def segment_tilestring(tilestring):
    # cached result is a simple, non-nested dictionary so this is safe
    return _cached_segment_tilestring(tilestring).copy()

@lru_cache(maxsize=int(1e6))
def tilestring_dims(tilestring):
    t = [r.strip() for r in tilestring.split('\n') if len(r.strip()) > 0]
    return len(t[0]), len(t)

def maze_code(maze):
    if isinstance(maze, str):
        string = '\n'.join([row.strip() for row in maze.strip().split('\n')])
    elif isinstance(maze, (list, tuple)):
        string = '\n'.join([row.strip() for row in maze])
    code = hashlib.blake2b(string.encode('utf-8'), digest_size=6).hexdigest()
    return code
