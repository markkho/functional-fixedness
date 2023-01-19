from types import SimpleNamespace
import numpy as np

def to_numpy(arr):
    return np.array([list(r) for r in arr])
def from_numpy(arr):
    return [''.join(r) for r in arr]

transform_grid = SimpleNamespace(
    base   = lambda g: [r for r in g],
    rot90  = lambda g: from_numpy(np.rot90(to_numpy(g), k=1)),
    rot180 = lambda g: from_numpy(np.rot90(to_numpy(g), k=2)),
    rot270 = lambda g: from_numpy(np.rot90(to_numpy(g), k=3)),
    vflip  = lambda g: from_numpy(np.flipud(to_numpy(g))),
    hflip  = lambda g: from_numpy(np.fliplr(to_numpy(g))),
    trans  = lambda g: from_numpy(np.transpose(to_numpy(g))),
    rtrans = lambda g: from_numpy(np.rot90(np.transpose(to_numpy(g)), k=2)),
)
transform_grid.base.inv   = transform_grid.base
transform_grid.rot90.inv  = transform_grid.rot270
transform_grid.rot180.inv = transform_grid.rot180
transform_grid.rot270.inv = transform_grid.rot90
transform_grid.vflip.inv  = transform_grid.vflip
transform_grid.hflip.inv  = transform_grid.hflip
transform_grid.trans.inv  = transform_grid.trans
transform_grid.rtrans.inv = transform_grid.rtrans

# Methods for transforming locations or points in a grid
def rotXY(deg, x, y, w, h, integer_state=True):
    rad = deg*np.pi/180
    rotMat = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    c = np.array([w/2, h/2])
    cXY = np.array([x, y]) - c + (.5 if integer_state else 0)
    nx, ny = rotMat@cXY + c - (.5 if integer_state else 0)
    return nx, ny

def vflipXY(x, y, w, h, integer_state=True):
    return x, -(y + (.5 if integer_state else 0) - h/2) + h/2 - (.5 if integer_state else 0)

def hflipXY(x, y, w, h, integer_state=True):
    return -(x + (.5 if integer_state else 0) - w/2) + w/2 - (.5 if integer_state else 0), y

def transXY(x, y, w, h, integer_state=True):
    nx, ny = vflipXY(x, y, w, h, integer_state=integer_state)
    return rotXY(270, nx, ny, w, h, integer_state=integer_state)

def rtransXY(x, y, w, h, integer_state=True):
    nx, ny = hflipXY(x, y, w, h, integer_state=integer_state)
    return rotXY(270, nx, ny, w, h, integer_state=integer_state)

# Transforming a discretized, integer-valued location where the origin is the lower-left cell
transform_location = SimpleNamespace(
    base= lambda g, s: s,
    rot90= lambda g, s: rotXY(90, s[0], s[1], len(g[0]), len(g), integer_state=True),
    rot180= lambda g, s: rotXY(180, s[0], s[1], len(g[0]), len(g), integer_state=True),
    rot270= lambda g, s: rotXY(270, s[0], s[1], len(g[0]), len(g), integer_state=True),
    vflip= lambda g, s: vflipXY(s[0], s[1], len(g[0]), len(g), integer_state=True),
    hflip= lambda g, s: hflipXY(s[0], s[1], len(g[0]), len(g), integer_state=True),
    trans= lambda g, s: transXY(s[0], s[1], len(g[0]), len(g), integer_state=True),
    rtrans= lambda g, s: rtransXY(s[0], s[1], len(g[0]), len(g), integer_state=True),
)
transform_location.base.inv   = transform_location.base
transform_location.rot90.inv  = transform_location.rot270
transform_location.rot180.inv = transform_location.rot180
transform_location.rot270.inv = transform_location.rot90
transform_location.vflip.inv  = transform_location.vflip
transform_location.hflip.inv  = transform_location.hflip
transform_location.trans.inv  = transform_location.trans
transform_location.rtrans.inv = transform_location.rtrans

# Transform a point where the origin is the lower-left corner
transform_point = SimpleNamespace(
    base= lambda g, s: s,
    rot90= lambda g, s: rotXY(90, s[0], s[1], len(g[0]), len(g), integer_state=False),
    rot180= lambda g, s: rotXY(180, s[0], s[1], len(g[0]), len(g), integer_state=False),
    rot270= lambda g, s: rotXY(270, s[0], s[1], len(g[0]), len(g), integer_state=False),
    vflip= lambda g, s: vflipXY(s[0], s[1], len(g[0]), len(g), integer_state=False),
    hflip= lambda g, s: hflipXY(s[0], s[1], len(g[0]), len(g), integer_state=False),
    trans= lambda g, s: transXY(s[0], s[1], len(g[0]), len(g), integer_state=False),
    rtrans= lambda g, s: rtransXY(s[0], s[1], len(g[0]), len(g), integer_state=False),
)
transform_point.base.inv   = transform_point.base
transform_point.rot90.inv  = transform_point.rot270
transform_point.rot180.inv = transform_point.rot180
transform_point.rot270.inv = transform_point.rot90
transform_point.vflip.inv  = transform_point.vflip
transform_point.hflip.inv  = transform_point.hflip
transform_point.trans.inv  = transform_point.trans
transform_point.rtrans.inv = transform_point.rtrans

# for transforming actions
transform_direction = SimpleNamespace(
    base = lambda a : a,
    rot90 = lambda a : tuple([int(i) for i in rotXY(90, *a, 0, 0, integer_state=False)]),
    rot180 = lambda a : tuple([int(i) for i in rotXY(180, *a, 0, 0, integer_state=False)]),
    rot270 = lambda a : tuple([int(i) for i in rotXY(270, *a, 0, 0, integer_state=False)]),
    vflip = lambda a : (a[0], -a[1]),
    hflip = lambda a : (-a[0], a[1]),
    trans= lambda a : (-a[1], -a[0]),
    rtrans= lambda a : (a[1], a[0]),
)
transform_direction.base.inv   = transform_direction.base
transform_direction.rot90.inv  = transform_direction.rot270
transform_direction.rot180.inv = transform_direction.rot180
transform_direction.rot270.inv = transform_direction.rot90
transform_direction.vflip.inv  = transform_direction.vflip
transform_direction.hflip.inv  = transform_direction.hflip
transform_direction.trans.inv  = transform_direction.trans
transform_direction.rtrans.inv = transform_direction.rtrans