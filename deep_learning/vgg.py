import torch
import torch.nn as nn


vgg_topologoes = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, topology, input_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.input_channels = input_channels
        self.convolutional_layers = self.create_conv_layers(topology)

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        return self.fully_connected(x)

    def create_conv_layers(self, architecture):
        layers = []
        input_channels = self.input_channels

        for x in architecture:
            if type(x) == int:
                output_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                input_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = VGG(
        topology=vgg_topologoes["VGG16"], input_channels=3, num_classes=1000
    ).to(device)
    batch_size = 3
    x = torch.randn(batch_size, 3, 224, 224).to(device)  # random single image
    assert model(x).shape == torch.Size([batch_size, 1000])
    print(model)
    print(model(x).shape)
