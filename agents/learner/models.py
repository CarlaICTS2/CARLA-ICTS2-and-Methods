import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.utils import Config
from utils.logger import log_info


class A2C(nn.Module):
    def __init__(self, hidden_dim, use_dropout):
        super(A2C, self).__init__()

        # use Akash Sinha's NavA2C architecture published here: https://arxiv.org/abs/2311.12875 or https://github.com/roboak/Nav-Q
        if Config.HyPLAN.NavA2C:
            # input_shape = [1, 3, 400, 400] = [batch_size, channels, width, height]
            self.feature_extractor = nn.Sequential(
                nn.LayerNorm([int(Config.Carla.SEGCAM_IMAGE_WIDTH), int(Config.Carla.SEGCAM_IMAGE_HEIGHT)]),
                # [1, 32, 99, 99]
                nn.Conv2d(3, 32, kernel_size=(8, 8), stride=(4, 4)),
                nn.ReLU(),
                # [1,64, 48, 48]
                nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
                nn.LayerNorm([64, 48, 48]),
                nn.ReLU(),
                # [1, 64, 46, 46]
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(),
                # [1, 128, 13, 13]
                nn.Conv2d(64, 128, kernel_size=(9, 9), stride=(3, 3)),
                nn.LayerNorm([128, 13, 13]),
                nn.ReLU(),
                # [1, 128, 5, 5]
                nn.Conv2d(128, 128, kernel_size=(9, 9), stride=(1, 1)),
                nn.ReLU(),
                # [1, hidden_dim, 1, 1]
                nn.Conv2d(128, hidden_dim, kernel_size=(5, 5), stride=(1, 1)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.LayerNorm([hidden_dim]),
                nn.ReLU()
            )

            self.memory = nn.LSTMCell(input_size=hidden_dim + 3, hidden_size=hidden_dim)

            self.actor = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm([64]),
                nn.ReLU(),
                nn.Linear(64, Config.Carla.NUM_ACTIONS)
            )

            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LayerNorm([64]),
                nn.ReLU(),
                nn.Dropout(p=0.5 if use_dropout else 0.0, inplace=False),
                nn.Linear(64, 1)
            )
        # use Florian Pusse's A2C architecture (originally proposed for HyLEAP)
        # published here: https://github.com/FlorianPusse/OpenDS-CTS
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.LazyLinear(out_features=hidden_dim),
                nn.ReLU()
            )

            self.memory = nn.LSTMCell(input_size=hidden_dim + 3, hidden_size=hidden_dim // 2)

            self.actor = nn.Linear(in_features=hidden_dim // 2, out_features=Config.Carla.NUM_ACTIONS)

            self.critic = nn.Sequential(
                nn.Dropout(p=0.5 if use_dropout else 0.0, inplace=False),
                nn.Linear(in_features=hidden_dim // 2, out_features=1)
            )

        self.sub_networks = [self.feature_extractor, self.memory, self.actor, self.critic]
        for sub_network in self.sub_networks:
            sub_network.apply(self.initialize_layer_weights)

    def initialize_layer_weights(self, layer):
        # initialization of NavA2C layers
        if isinstance(layer, nn.Conv2d) and Config.HyPLAN.NavA2C:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))

        # different initialization of FFC layers for A2C and NavA2C
        if isinstance(layer, nn.Linear) and not isinstance(layer, nn.Linear):
            if Config.HyPLAN.NavA2C:
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
            else:
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                nn.init.zero_(layer.bias)

        if isinstance(layer, nn.LSTM) and Config.HyPLAN.NavA2C:
            nn.init.zero_(layer.bias)

    def _keep_dropout_active(self, layer):
        if isinstance(layer, nn.Dropout):
            layer.train()

    def keep_dropout_active(self):
        self.critic.apply(self._keep_dropout_active)

    def forward(self, car_intention, previous_lstm_hidden_state, previous_lstm_cell_state, provided_features):
        extracted_features = self.feature_extractor(car_intention)
        all_features = torch.cat((extracted_features, provided_features), dim=-1)
        current_lstm_hidden_state, current_lstm_cell_state = self.memory(all_features,
                                                                         (previous_lstm_hidden_state,
                                                                          previous_lstm_cell_state))
        policy = self.actor(current_lstm_hidden_state)
        value = self.critic(current_lstm_hidden_state)

        return policy, value, (current_lstm_hidden_state, current_lstm_cell_state)

    # split forward pass in order to minimize computational redundancy when obtaining uncertainty values
    def forward_feature_extractor(self, car_intention, previous_lstm_hidden_state, previous_lstm_cell_state,
                                  provided_features):
        extracted_features = self.feature_extractor(car_intention)
        all_features = torch.cat((extracted_features, provided_features), dim=-1)
        current_lstm_hidden_state, current_lstm_cell_state = self.memory(all_features,
                                                                         (previous_lstm_hidden_state,
                                                                          previous_lstm_cell_state))
        return current_lstm_hidden_state, current_lstm_cell_state

    def forward_actor(self, current_lstm_hidden_state):
        return self.actor(current_lstm_hidden_state)

    # only the forward pass of the critic has to be executed multiple times,
    # because it is the only sub-network that has a dropout layer
    def forward_critic(self, current_lstm_hidden_state):
        return self.critic(current_lstm_hidden_state)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=True)


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).apply(self.initialize_weights_he)

    def initialize_weights_he(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, states):
        states = states.permute(0, 3, 1, 2)
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(46 * 46 * 64 + 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, probs=None, steps=None):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        if probs is not None and steps is not None:
            action_probs = probs
            # if steps < Config.pre_train_steps:
            #     action_probs = probs
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs