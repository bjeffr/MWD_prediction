from torch import nn


class NN_Module_1C(nn.Module):
    def __init__(self, num_targets=2, num_features=140):
        super(NN_Module_1C, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
        )
        self.hidden_stack = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.output = nn.Linear(256, num_targets)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_stack(x)
        return self.output(x)


class NN_Module_2C(nn.Module):
    def __init__(self, num_targets=5, num_features=140):
        super(NN_Module_2C, self).__init__()
        self.input = nn.Unflatten(
            dim=1, unflattened_size=(1, 2, int(num_features / 2))
        )
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(32 * num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.output = nn.Linear(128, num_targets)

    def forward(self, x):
        x = self.input(x)
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        return self.output(x)
