import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, topology):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(topology) - 1):
            layers += (
                [nn.Linear(topology[i], topology[i + 1]), nn.ReLU()]
                if i != len(topology) - 2
                else [nn.Linear(topology[i], topology[i + 1])]
            )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":
    topology = [4, 16, 32, 16, 3]
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = MLP(topology=topology).to(device)
    print(model)
