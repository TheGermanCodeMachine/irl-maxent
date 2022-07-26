#!/usr/bin/env python

from ftfy import no_type_check
from irl_maxent import maxent as M
from irl_maxent import plot as P
from irl_maxent import trajectory as T
from irl_maxent import solver as S
from irl_maxent import optimizer as O
from irl_maxent import my_mdp as MDP
from irl_maxent import mdp_false_state_belief as FSMDP
from irl_maxent.trajectory import Trajectory

import numpy as np
import matplotlib.pyplot as plt
import random

# example running the MDP in: https://www.alignmentforum.org/s/4dHMdK5TLN6xcqtyc/p/cnC2RMWEGiGpJv8go#Time_inconsistency_and_Procrastination

def setup_false_mdp():
    """
    Set-up our MDP/GridWorld
    """

    size = 3
    R = 10
    w = 2
    eps = 1
    # create our world
    false_world = FSMDP.FalseStateBeliefMDP(size=4)

    # set up the reward function
    false_reward = np.zeros(false_world.n_states)
    false_reward[0] = 2*R
    false_reward[3] = R

    # set up terminal states
    false_terminal = [0,3]
    return false_world, false_reward, false_terminal

def setup_true_mdp():
    """
    Set-up our MDP/GridWorld
    """

    size = 3
    R = 10
    w = 2
    eps = 1
    # create our world
    true_world = FSMDP.FalseStateBeliefMDP(size=3)

    # set up the reward function
    true_reward = np.zeros(true_world.n_states)
    true_reward[2] = R

    # set up terminal states
    true_terminal = [2]
    return true_world, true_reward, true_terminal


def generate_trajectories(world, reward, terminal, start, n_trajectories):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    discount = 0.7
    weighting = lambda x: x**5

    # set up initial probabilities for trajectory generation
    initial = np.zeros((n_trajectories, world.n_states))
    for i, j in enumerate(initial):
        initial[i, start[i]] = 1.0

    # generate trajectories
    value = S.value_iteration(world.p_transition, reward, discount)
    policy = S.optimal_policy_from_value(world, value)
    policy_exec = T.policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, initial, terminal))

    return tjs, policy


def maxent(world, terminal, trajectories):
    """
    Maximum Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = MDP.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl(world.p_transition, features, terminal, trajectories, optim, init)

    return reward


def maxent_causal(world, terminal, trajectories, discount=0.7):
    """
    Maximum Causal Entropy Inverse Reinforcement Learning
    """
    # set up features: we use one feature vector per state
    features = MDP.state_features(world)

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = M.irl_causal(world.p_transition, features, terminal, trajectories, optim, init, discount)

    return reward

def run():
    # common style arguments for plotting
    style = {
        'border': {'color': 'red', 'linewidth': 0.5},
    }

    n_trajectories = 200

    # set-up mdp
    false_world, false_reward, false_terminal = setup_false_mdp()
    true_world, true_reward, true_terminal = setup_true_mdp()

    # show our original reward
    ax = plt.figure(num='Original Reward').add_subplot(111)
    P.plot_state_values(ax, true_world, true_reward, **style)
    plt.draw()

    # generate "expert" trajectories
    starts = np.repeat(2, n_trajectories)
    false_trajectories, false_expert_policy = generate_trajectories(false_world, false_reward, false_terminal, start=starts, n_trajectories=n_trajectories)

    # cut off at trajectories at some point and record exit points
    exit_points = []
    cut_trajectories = []
    for ft in false_trajectories:
        cut_trajectory = []
        for trans in ft.transitions():
            if trans[2] == 0: # if the next transition goes into a non-existing state cut it there and record cut-off point
                cut_trajectories.append(cut_trajectory)
                exit_points.append(trans[0])
            else: # otherwise just leave the transition as is
                cut_trajectory.append(trans)

    exit_points = [i-1 for i in exit_points]

    true_trajectories, true_expert_policy = generate_trajectories(true_world, true_reward, true_terminal, start=exit_points, n_trajectories=n_trajectories)
    full_trajectories = []
    for i in range(n_trajectories):
        cut_trajectories[i].extend(true_trajectories[i].transitions())
        full_trajectories.append(Trajectory(cut_trajectories[i]))

    # show our expert policies
    ax = plt.figure(num='Expert Trajectories and Policy').add_subplot(111)
    P.plot_deterministic_policy(ax, true_world, false_expert_policy, **style)

    for t in full_trajectories:
        P.plot_trajectory(ax, true_world, t, lw=5, color='white', alpha=0.025)

    plt.draw()

    # maximum entropy reinforcement learning (non-causal)
    reward_maxent = maxent(true_world, true_terminal, full_trajectories)

    # show the computed reward
    ax = plt.figure(num='MaxEnt Reward').add_subplot(111)
    P.plot_state_values(ax, true_world, reward_maxent, **style)
    plt.draw()

    # maximum casal entropy reinforcement learning (non-causal)
    # reward_maxcausal = maxent_causal(world, terminal, trajectories)

    # show the computed reward
    # ax = plt.figure(num='MaxEnt Reward (Causal)').add_subplot(111)
    # P.plot_state_values(ax, world, reward_maxcausal, **style)
    # plt.draw()

    # plt.show()

    print("false belief", false_reward)
    print("false_expert_policy", ['l' if false_expert_policy[i]==0 else 'r' for i in range(false_world.n_states)])
    print("true world", true_reward)
    print("true_expert_policy", ['l' if true_expert_policy[i]==0 else 'r' for i in range(true_world.n_states)])
    print("true world", true_reward)
    print("derived_state_values", [reward_maxent[i] for i in range(true_world.n_states)])
    print("example trajectory", ['l' if trans[1]==0 else 'r' for trans in random.choice(full_trajectories).transitions()])
