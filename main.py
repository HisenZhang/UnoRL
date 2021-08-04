import numpy as np
import random
import time


class QAgent(object):
    """
    Implementation of a Q-learning Algorithm
    """

    def __init__(self, env, name, state_size, action_size, learning_parameters, exploration_parameters):
        """
        initialize the q-learning agent
        Args:
          name (str): set the name of the Q-Agent
          state_size (int): ..
          action_size (int): ..
          learning_parameters (dict): 
          exploration_parameters (dict):

        """
        self.name = name

        # init the Q-table
        self.qtable = np.zeros((state_size, action_size))
        self.result = np.zeros((state_size, action_size))

        # learning parameters
        self.learning_rate = learning_parameters['learning_rate']
        self.gamma = learning_parameters['gamma']

        # exploration parameters
        self.epsilon = exploration_parameters['epsilon']
        self.max_epsilon = exploration_parameters['max_epsilon']
        self.min_epsilon = exploration_parameters['min_epsilon']
        self.decay_rate = exploration_parameters['decay_rate']

        self.env = env

    def q_learning(self, plot=False, max_steps=10, total_episodes=1000):
        """
        implementation of the q-learning algorithm, here the q-table values are calculated
        Args:
          plot (boolean): set true, to get trainings progress 
          max_steps (int): number of stepts an agent can take, before the environment is reset 
          total_episodes (int): total of training episodes (the number of trials a agent can do)          
        """

        # create placeholders to store the results
        self.episode_rewards = np.zeros(total_episodes)
        self.episode_epsilon = np.zeros(total_episodes)
        self.episode_last_state = np.zeros(total_episodes)

        start = time.time()
        # loop over all episodes
        for episode_i in range(total_episodes):
            # initalize the environment
            state = self.env.reset()

            # for each episode loop over the max number of steps that are possible
            for step in range(max_steps):

                # get action, e-greedy
                action = self.get_action(state)

                # take an action and observe the outcome state (new_state), reward and stopping criterion
                new_state, reward, done, _ = self.env.step(action)

                self.qtable[state, action] = self.update_qtable(
                    state, new_state, action, reward, done)
                state = new_state
                self.episode_rewards[episode_i] += reward
                self.result = np.dstack((self.result, self.qtable))

                # check stopping criterion
                if done == True:
                    break

            self.episode_rewards[episode_i] /= step  # average the reward
            self.episode_last_state[episode_i] = state  # average the reward

            # reduce epsilon, for exploration-exploitation tradeoff
            self.update_epsilon(episode_i)

            if episode_i % 100 == 0 and plot:
                print('episode: {}'.format(episode_i))
                print('\telapsed time [min]: {} reward {}'.format(
                    round((time.time() - start)/60, 1), self.episode_rewards[episode_i]))

    def update_qtable(self, state, new_state, action, reward, done):
        """
        update the q-table: Q(s,a) = Q(s,a) + lr  * [R(s,a) + gamma * max Q(s',a') - Q (s,a)]
        Args:
          state (int): current state of the environment
          new_state (int): new state of the environment
          action (int): current action taken by agent
          reward (int): current reward received from env
          done (boolean): variable indicating if env is done
        Returns:
          qtable (array): the qtable containing a value for every state (y-axis) and action (x-axis) 
        """
        return self.qtable[state, action] + self.learning_rate * \
            (reward + self.gamma *
             np.max(self.qtable[new_state, :]) * (1 - done) - self.qtable[state, action])

    def update_epsilon(self, episode):
        """
        reduce epsilon, exponential decay
        Args:
          episode (int): number of episode
        """
        self.epsilon = self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.decay_rate*episode)

    def get_action(self, state):
        """
        select action e-greedy
        Args:
          state (int): current state of the environment/agent
        Returns:
          action (int): action that the agent will take in the next step
        """
        if random.uniform(0, 1) >= self.epsilon:
            # exploitation, max value for given state
            action = np.argmax(self.qtable[state, :])
        else:
            # exploration, random choice
            action = self.env.action_space.sample()
        return action
