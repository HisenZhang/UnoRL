import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from collections import namedtuple

Transition = namedtuple('Transition', [
                        'state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])


class SARSAAgent():
    def __init__(self,
                 update_target_estimator_every=1000,
                 replay_memory_size=20000,
                 batch_size=32,
                 dr=0.95,
                 lr=0.01,
                 num_actions=0,
                 state_shape=0,
                 mlp_layers=None,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=2000):
        self.use_raw = False
        self.dr = dr
        self.lr = lr
        self.epsilons = np.linspace(
            epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.num_actions = num_actions
        self.update_target_estimator_every = update_target_estimator_every

        self.t = 0

        self.memory_size = replay_memory_size
        self.batch_size = batch_size
        self.memory = []

        self.net = SarsaNet(lr=self.lr, mlp_layers=mlp_layers,
                            state_shape=state_shape, num_actions=num_actions)
        self.target_net = deepcopy(self.net)
        self.train_t = 0

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state['obs'], action, reward, next_state['obs'], list(
            next_state['legal_actions'].keys()), done)
        self.memory.append(transition)

        self.t += 1
        if len(self.memory) >= self.batch_size:
            self.train()

    def step(self, state):
        q_values = self.predict(state)

        epsilon = self.epsilons[min(self.t, self.epsilon_decay_steps-1)]

        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) == 1:
            action = legal_actions[0]
        elif random.uniform(0.0, 1.0) < epsilon:

            legal_actions.remove(np.argmax(q_values))

            action = random.choice(legal_actions)
        else:
            action = np.argmax(q_values)

        return action

    def eval_step(self, state):
        q_values = self.predict(state)
        best_action = np.argmax(q_values)

        info = {}
        # info['values'] = {state['raw_legal_actions'][0]: float(q_values[list(state['legal_actions'].keys())[0]]) for i in range(len(state['legal_actions']))}
        # info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        q_predictions = self.net.get_predictions(
            np.expand_dims(state['obs'], 0))[0]

        q_values = [q_predictions[i] if i in state['legal_actions']
                    else -np.inf for i in range(self.num_actions)]

        # print("prediction:", q_predictions )
        # print("q_vals:",q_values)
        # for i in range(len(q_values)):
        #   if(q_values[i] != -np.inf):
        #     print("i:", i, ", q_value:", q_values[i])
        # print("best:", np.argmax(q_values))
        return q_values

    def train(self):
        # print("training...")
        samples = random.sample(self.memory, self.batch_size)
        # samples = random.sample(self.memory, 2)
        state_batch, action_batch, reward_batch, next_state_batch, legal_next_actions_batch, done_batch = map(
            np.array, zip(*samples))
        # self.net.get_predictions
        # print("prediction:". self.net.get_predictions(np.expand_dims(state['obs'], 0))[0])
        # print("samples:", samples)
        # print("states:", state_batch)
        # print("a:", action_batch)
        # print("l:", legal_next_actions_batch)

        # gets q' from s'
        q_next = self.net.get_predictions(next_state_batch)
        q_next_mask = [[q_next[j][i] if i in legal_next_actions_batch[j] else -
                        np.inf for i in range(self.num_actions)] for j in range(self.batch_size)]
        q_next_mask = np.array(q_next_mask)

        # uses q' to find a'
        epsilon = self.epsilons[min(self.t, self.epsilon_decay_steps-1)]
        next_actions = []
        for i in range(self.batch_size):
            action = np.argmax(q_next_mask[i])

            if random.uniform(0.0, 1.0) < epsilon and len(legal_next_actions_batch[i]) > 1:
                other_actions = deepcopy(legal_next_actions_batch[i])
                other_actions.remove(action)
                action = random.choice(other_actions)

            next_actions.append(action)
        next_actions = np.array(next_actions)
        q_next_target = self.net.get_predictions(next_state_batch)
        q_next_target_mask = [[q_next_target[j][i] for i in range(
            self.num_actions)] for j in range(self.batch_size)]
        q_next_target_mask = np.array(q_next_target_mask)
        target_batch = reward_batch + np.invert(done_batch).astype(
            np.float32) * self.dr * q_next_target_mask[np.arange(self.batch_size), next_actions]

        loss = self.net.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.t, loss), end='')

        if self.train_t % self.update_target_estimator_every == 0:
            self.target_net = deepcopy(self.net)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1


class SarsaNet(nn.Module):
    def __init__(self, lr=0.01, num_actions=0, state_shape=0, mlp_layers=None):

        super(SarsaNet, self).__init__()
        self.lr = lr
        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        layer_dims = [np.prod(self.state_shape[0])] + self.mlp_layers
        model = []
        # model.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            model.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            model.append(nn.Tanh())
        model.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))

        self.model = nn.Sequential(*model)

        for p in self.model.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, s):

        # q_predictions = [random.uniform(-1.0, 1.0) for _ in range(self.num_actions)]
        with torch.no_grad():
            q_predictions = self.model(s)
        return q_predictions

    def get_predictions(self, states):
        states = torch.from_numpy(states).float()
        q_vals = []
        for s in states:
            q_vals.append(self.forward(torch.flatten(s)))
        return q_vals

    def update(self, states, actions, targets):

        self.optimizer.zero_grad()
        self.model.train()

        s = torch.from_numpy(states).float()
        a = torch.from_numpy(actions).long()
        y = torch.from_numpy(targets).float()

        # print("s:", s)
        # print("sf:" , torch.flatten(s, start_dim=1))
        all_q = self.model(torch.flatten(s, start_dim=1))
        # print("all_q:", all_q)

        q_actual = torch.gather(
            all_q, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # print("q_actual:", q_actual)

        # update model
        batch_loss = self.mse_loss(q_actual, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.model.eval()

        return batch_loss
