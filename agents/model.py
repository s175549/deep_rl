import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, alpha, num_frames):
        super().__init__()
        torch.manual_seed(64)
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[1])
        self.fc4 = nn.Linear(hidden_size[1], output_size)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.8)
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x