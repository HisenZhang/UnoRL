# !pip3 install rlcard[torch]

import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import namedtuple

import rlcard
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

import matplotlib
import matplotlib.pyplot as plt

from agent.SARSA import SARSAAgent, SarsaNet


# hyperparameters
episode_count = 10000
EPS_STEP = 100
ALPHA = .00005  # .5
GAMMA = .99
eval_freq = 100
eval_games = 50


temp_env = rlcard.make('uno')
uno_num_actions = temp_env.num_actions
uno_state_shape = temp_env.state_shape


"""# Training Loop"""


def train(num_players, training_agents):

    # makes sure num_players and agents are valid
    if len(training_agents) > num_players or num_players == 1:
        print("invalid input")
        return

    agents = training_agents.copy()

    # creates random agents to fill in empty player slots
    initial_agents = len(agents)
    for _ in range(num_players-initial_agents):
        agents.append(RandomAgent(num_actions=uno_num_actions))

    # creates the environment for training
    env = rlcard.make('uno')
    # print("agents:", agents)
    env.set_agents(agents)

    performances = [[]] * len(training_agents)
    for e in range(episode_count):
        trajectories, payoffs = env.run(is_training=True)

        data = reorganize(trajectories, payoffs)
        # print("data:",data[0][0])

        # data is a list in the form (state, action, reward, next_state, done)
        for i in range(len(training_agents)):
            for hand in data[i]:
                # return hand
                training_agents[i].feed(hand)

        if e % eval_freq == 0:
            eval_results = tournament(env, eval_games)
            for i in range(len(training_agents)):
                performances[i].append(eval_results[i])
                m = [m*eval_freq for m in range(len(performances[i]))]
                plt.plot(performances[i])
                # plt.legend()
                plt.savefig('data33/{}.png'.format(e))
                print(" Write to \"data33/{}.png\"".format(e))
    return performances


"""# Agent Maker and Results"""

dqn_agent = DQNAgent(num_actions=uno_num_actions,
                     state_shape=uno_state_shape[0],
                     mlp_layers=[512, 1024, 512],
                     device=get_device(),
                     epsilon_decay_steps=EPS_STEP,
                     discount_factor=GAMMA,
                     learning_rate=ALPHA
                     )

sarsa_agent = SARSAAgent(dr=GAMMA,
                         lr=ALPHA,
                         mlp_layers=[64, 128, 64],
                         num_actions=uno_num_actions,
                         state_shape=uno_state_shape[0],
                         epsilon_decay_steps=EPS_STEP)
m = None

dqn_performance = train(2, [dqn_agent])[0]
# sarsa_performance = train(2, [sarsa_agent])[0]

# for i in range(1, 100000, 5):
#     for j in range(90, 100, 1):
#         ALPHA = i / 100000
#         GAMMA = j / 100
#         print(ALPHA,GAMMA)
#         train(2, [dqn_agent])[0]
#         train(2, [sarsa_agent])[0]
