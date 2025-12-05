import torch.nn as nn
 
class SimpleModel(nn.Module):
    """
    Small MLP used for demos.
    """
    def __init__(self, input_size: int, output_size: int):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)
