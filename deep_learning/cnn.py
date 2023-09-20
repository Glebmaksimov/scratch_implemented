import torch.nn.functional as F
import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, input_shape=1, num_classes=10):
        super().__init__()

        self.convolutional_layer_1 = nn.Conv2d(
            in_channels=input_shape,
            out_channels=8,  # number if convolutional layer output color cnannels(feature maps)
            kernel_size=(3, 3),  # filter matrix size
            stride=(1, 1),  # filter matrix step size
            padding=(1, 1),
        )  # we need padding to restrict filter miving.Padding added to all six sides of input.
        # bacause it can lose the information due to the full ramge of motion.

        self.pooling_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.convolutional_layer_2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),  # x.reshape(x.shape[0],-1)
            nn.Linear(in_features=16 * 7 * 7, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = F.relu(self.convolutional_layer_1(x))
        print("after 1st convolution shape: ", x.shape)
        x = self.pooling_layer(x)
        print("after 1st pooling shape: ", x.shape)
        x = F.relu(self.convolutional_layer_2(x))
        print("after 2nd convolution shape: ", x.shape)
        x = self.pooling_layer(x)
        print("after 2nd pooling shape: ", x.shape)
        return self.fully_connected_layer(x)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = CNN().to(device)
    x = torch.rand((1, 1, 28, 28)).to(device)
    print(model)
    print(model(x).shape)
