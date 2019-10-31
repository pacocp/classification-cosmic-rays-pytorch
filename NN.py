import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# seed for weight initialization
torch.manual_seed(1234)


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, units, output_size):
        super(NeuralNet, self).__init__()
        self.input = nn.Linear(input_size, units[0])
        self.relu = nn.ReLU()
        self.fcs = []
        if(len(units) > 1):
            for i in range(0, len(units)-1):
                self.fcs.append(nn.Linear(units[i], units[i+1]))
        self.output = nn.Linear(units[-1], output_size)

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        if(len(self.fcs) >= 1):
            for f in self.fcs:
                out = f(out)
                out = self.relu(out)
        out = self.output(out)
        return out


# 1D convolutional neural network
class ConvNeuralNet(nn.Module):
    def __init__(self, input_size, units, output_size):
        super(ConvNeuralNet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(1, units[0], kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1)
        )
        self.hiddens = []
        if(len(units) > 1):
            for i in range(0, len(units)-1):
                hidden = nn.Sequential(
                            nn.Conv1d(units[i], units[i+1], kernel_size=10),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=1, stride=1)
                            )
                self.hiddens.append(hidden)

        self.units = units
        
    def forward(self, x):
        out = self.input(x)
        if(len(self.hiddens) >= 1):
            i = 1
            for h in self.hiddens:
                out = h(out)
                i += 1
        out_ch = out.size(2)
        out = out.view(out.size(0), -1)
        output = nn.Sequential(
            nn.Linear(self.units[-1]*out_ch, 5)
        )
        out = output(out)
        return out


# 1D convolutional neural network
class ConvNeuralNet2Dense(nn.Module):
    def __init__(self, input_channels, units, output_size):
        super(ConvNeuralNet2Dense, self).__init__()
        self.input = nn.Sequential(
            nn.Conv1d(input_channels, units[0], kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        self.h1 = nn.Sequential(
                            nn.Conv1d(units[0], units[1], kernel_size=5, 
                                      stride=1),
                            nn.ReLU(),
                            nn.MaxPool1d(kernel_size=2))
        self.fc1 = nn.Sequential(
            nn.Linear(units[1]*47, units[2]),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
                    nn.Linear(units[2], units[3]),
                    nn.ReLU()
                )
                
        self.output = nn.Sequential(
            nn.Linear(units[3], output_size)
        )
        self.units = units
        
    def forward(self, x):
        out = self.input(x)
        out = self.h1(out)
        # out_ch = out.size(2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.output(out)
        return out
