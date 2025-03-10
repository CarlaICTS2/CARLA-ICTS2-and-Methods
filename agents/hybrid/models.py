import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import log_debug
from utils.utils import normalized_columns_initializer


#=================================================================================#
#                                   LEADER                                        #
#=================================================================================#
class AttentionNet(nn.Module):
    def __init__(self, input_size, is_weight_size, gru_size):
        super(AttentionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 512)
        self.fc8 = nn.Linear(512, gru_size)
        self.gru = nn.GRUCell(gru_size, gru_size)
        self.fc9 = nn.Linear(gru_size, 512)
        self.fc10_mean = nn.Linear(512, is_weight_size)
        self.fc10_std = nn.Linear(512, is_weight_size)
        log_debug((f"initialized AttentionNet with input size {input_size}, " +\
                   f"gru size {gru_size} and attention size {is_weight_size}"))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data = normalized_columns_initializer(layer.weight.data.size(), 0.5)
                layer.bias.data.fill_(0)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)
        torch.nn.init.constant_(self.fc10_std.bias, 1)

    def forward(self, data, h):
        x = F.relu(self.fc1(data))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        grux = self.gru(x, h)
        x = F.relu(self.fc9(grux))
        return F.softmax(self.fc10_mean(x)), grux


class CriticNet(nn.Module):
    def __init__(self, input_size, gru_size):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, gru_size)
        self.gru = nn.GRUCell(gru_size, gru_size)
        self.fc4 = nn.Linear(gru_size, 256)
        self.fc5 = nn.Linear(256, 32)
        self.fc6 = nn.Linear(32, 1)
        log_debug(f"initialized CriticNet with input size {input_size} and gru size {gru_size}")

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data = normalized_columns_initializer(layer.weight.data.size(), 0.5)
                layer.bias.data.fill_(0)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, data, h):
        x = F.relu(self.fc1(data))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        grux = self.gru(x, h)
        x = F.relu(self.fc4(grux))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x, grux
