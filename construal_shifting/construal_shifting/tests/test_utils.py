from construal_shifting.gridutils import transform_direction

def test_direction_transform():
    action_transform_test_cases = dict(
        base=[],
        rot90=[((1, 0), (0, 1)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)), ((0, -1), (1, 0))],
        rot180=[((1, 0), (-1, 0)), ((-1, 0), (1, 0)), ((0, 1), (0, -1)), ((0, -1), (0, 1))],
        rot270=[((1, 0), (0, -1)), ((-1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))],
        vflip =[((1, 0), (1, 0)), ((-1, 0), (-1, 0)), ((0, 1), (0, -1)), ((0, -1), (0, 1))],
        hflip =[((1, 0), (-1, 0)), ((-1, 0), (1, 0)), ((0, 1), (0, 1)), ((0, -1), (0, -1))],
        trans= [((1, 0), (0, -1)), ((-1, 0), (0, 1)), ((0, 1), (-1, 0)), ((0, -1), (1, 0)), ((1, 1), (-1, -1))],
        rtrans= [((1, 0), (0, 1)), ((-1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0)), ((1, 1), (1, 1))],
    )
    for tname, cases in action_transform_test_cases.items():
        for a, exp in cases:
            assert getattr(transform_direction, tname)(a) == exp
            assert getattr(transform_direction, tname).inv(exp) == a