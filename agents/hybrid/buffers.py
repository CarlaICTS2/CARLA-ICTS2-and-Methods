import numpy as np


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def append(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            del self.buffer[:-self.capacity]

    def extend(self, items):
        self.buffer.extend(items)
        if len(self.buffer) > self.capacity:
            del self.buffer[:-self.capacity]

    def sample(self, count):
        return [self.buffer[i] for i in np.random.choice(np.arange(len(self.buffer)), count)]


class ExperienceBuffer:
    
    def __init__(self, buffer_size=50):
        self.buffer = []
        self.buffer_size = buffer_size
        self.num_entries = 0

    def num_entries(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self):
        return self.buffer[np.random.randint(0, len(self.buffer))]
