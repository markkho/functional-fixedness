from itertools import product

import numpy as np

from msdm.core.table import Table
from msdm.core.table.tableindex import TableIndex, domaintuple
from construal_shifting.task_modeling.participant_model import ParticipantSetHiddenMarkovModel

def test_ParticipantSetHiddenMarkovModel():
    class TestHMM(ParticipantSetHiddenMarkovModel):
        set_transition_matrix = Table(
            data=np.array([
                [
                    [.7, .3],
                    [.3, .7],
                ],
                [
                    [.72, .28],
                    [.27, .73],
                ],
                [
                    [.69, .31],
                    [.32, .68],
                ],
            ]),
            table_index=TableIndex(
                field_names=['timestep', 'last_state', 'state'],
                field_domains=[
                    domaintuple([0, 1, 2]),
                    domaintuple(['rain', 'no_rain']),
                    domaintuple(['rain', 'no_rain'])
                ]
            )
        )
        set_likelihood_matrix = Table(
            data=np.array([
                [.9, .2], #umbrella
                [.1, .8], #no umbrella
                [.1, .8], #no umbrella
            ]),
            table_index=TableIndex(
                field_names=['timestep', 'state'],
                field_domains=[
                    domaintuple([0, 1, 2]),
                    domaintuple(['rain', 'no_rain']),
                ]
            )
        )
        def __init__(self, initial_sets = (.5, .5)):
            super().__init__(None, None, initial_sets)

    hmm = TestHMM()

    #forward pass
    forward_0 = hmm.forward(0)
    exp_forward_0 = np.array([.9*(.7*.5+.3*.5), .2*(.3*.5+.7*.5)])
    exp_forward_0 /= exp_forward_0.sum()
    assert np.isclose(forward_0, exp_forward_0).all()

    forward_1 = hmm.forward(1)
    exp_forward_1 = np.array([
        .1*(.72*exp_forward_0[0] + .27*exp_forward_0[1]),
        .8*(.28*exp_forward_0[0] + .73*exp_forward_0[1]),
    ])
    exp_forward_1 /= exp_forward_1.sum()
    assert np.isclose(forward_1, exp_forward_1).all()

    forward_2 = hmm.forward(2)
    exp_forward_2 = np.array([
        .1*(.69*exp_forward_1[0] + .32*exp_forward_1[1]),
        .8*(.31*exp_forward_1[0] + .68*exp_forward_1[1]),
    ])
    exp_forward_2 /= exp_forward_2.sum()
    assert np.isclose(forward_2, exp_forward_2).all()

    # backward pass (in log prob)
    backward_2 = hmm.backward_log(2)
    exp_backward_2 = np.log([1, 1])
    assert np.isclose(backward_2, exp_backward_2).all()

    backward_1 = hmm.backward_log(1)
    exp_backward_1 = np.log([
        .69*.1 + .31*.8,
        .32*.1 + .68*.8
    ])
    assert np.isclose(backward_1, exp_backward_1).all()

    backward_0 = hmm.backward_log(0)
    exp_backward_0 = np.log([
        .72*.1*np.exp(exp_backward_1[0]) + .28*.8*np.exp(exp_backward_1[1]),
        .27*.1*np.exp(exp_backward_1[0]) + .73*.8*np.exp(exp_backward_1[1])
    ])
    assert np.isclose(backward_0, exp_backward_0).all()

    # state marginals
    marg_0 = hmm.state_marginal(0)
    exp_marg_0 = exp_forward_0*np.exp(exp_backward_0)
    exp_marg_0 /= exp_marg_0.sum()
    assert np.isclose(exp_marg_0, marg_0).all()

    marg_1 = hmm.state_marginal(1)
    exp_marg_1 = exp_forward_1*np.exp(exp_backward_1)
    exp_marg_1 /= exp_marg_1.sum()
    assert np.isclose(exp_marg_1, marg_1).all()

    marg_2 = hmm.state_marginal(2)
    exp_marg_2 = exp_forward_2*np.exp(exp_backward_2)
    exp_marg_2 /= exp_marg_2.sum()
    assert np.isclose(exp_marg_2, marg_2).all()

    exp_marg = np.stack([exp_marg_0, exp_marg_1, exp_marg_2])
    assert np.isclose(hmm.state_marginals(), exp_marg).all()

    # trajectory marginals
    exp_traj_marg = {
        (0, 1): np.array([
            [
                np.exp(backward_1[0])*.1*.72*forward_0[0], # rain, rain
                np.exp(backward_1[1])*.8*.28*forward_0[0], # rain, no rain
            ],
            [
                np.exp(backward_1[0])*.1*.27*forward_0[1], # no rain, rain
                np.exp(backward_1[1])*.8*.73*forward_0[1], # no rain, no rain
            ]
        ]), 
        (1, 2): np.array([
            [
                np.exp(backward_2[0])*.1*.69*forward_1[0], # rain, rain
                np.exp(backward_2[1])*.8*.31*forward_1[0], # rain, no rain
            ],
            [
                np.exp(backward_2[0])*.1*.32*forward_1[1], # no rain, rain
                np.exp(backward_2[1])*.8*.68*forward_1[1], # no rain, no rain
            ]
        ]), 
    }
    exp_traj_marg[(0, 1)] /= exp_traj_marg[(0, 1)].sum()
    exp_traj_marg[(1, 2)] /= exp_traj_marg[(1, 2)].sum()
    assert np.isclose(exp_traj_marg[(0, 1)], hmm.trajectory_marginal(0, 1)).all()
    assert np.isclose(exp_traj_marg[(1, 2)], hmm.trajectory_marginal(1, 2)).all()

    markov_likelihood = np.array(hmm.set_transition_matrix)*np.array(hmm.set_likelihood_matrix)[:, None, :]
    full_traj_likelihood = np.zeros((2, 2, 2))
    for r_start, r0, r1, r2 in product((0, 1), (0, 1), (0, 1), (0, 1)):
        traj_prob = \
            .5*markov_likelihood[0, r_start, r0]*\
            markov_likelihood[1, r0, r1]*\
            markov_likelihood[2, r1, r2]
        full_traj_likelihood[r0, r1, r2] += traj_prob
    full_traj_likelihood /= full_traj_likelihood.sum()
    full_traj_likelihood = full_traj_likelihood.reshape(2, -1)
    assert np.isclose(full_traj_likelihood, hmm.trajectory_marginal(0, 2)).all()

    # test consistency
    assert np.isclose(np.array(hmm.trajectory_marginal(0, 2)).sum(-1), hmm.state_marginal(0)).all()
    assert np.isclose(np.array(hmm.trajectory_marginal(1, 2)).sum(-1), hmm.state_marginal(1)).all()
    assert np.isclose(np.array(hmm.trajectory_marginal(2, 2)).sum(-1), hmm.state_marginal(2)).all()