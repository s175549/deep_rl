import random
from collections import deque

class ReplayBuffer(object):

    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)
        self.transition = transition
        self.seed = random.seed(1234)

    def store(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size, rand=True):
        if rand == True:
            return random.sample(self.memory, batch_size)
        else:
            samples = [self.memory[j] for j in range(batch_size)]
            return samples

    def clear(self):
        self.__init__(self.capacity,self.transition)

    def size(self):
        return len(self.memory)