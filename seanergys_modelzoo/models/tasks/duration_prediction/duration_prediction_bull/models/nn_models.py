"""
PyTorch neural network architectures used in the duration prediction pipeline.

Includes:
- MLP (fully connected)
- CNN (1D convolution + dense layers)
- MLP with dropout regularization

All models are designed to be lightweight and compatible with sklearn-style wrappers.
"""

from typing import List, Union

import numpy as np
import torch


class LinearActivation(torch.nn.Module):
    """
    Identity activation function.

    This module returns the input tensor unchanged. It is useful as a placeholder
    when no non-linearity is desired.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Same tensor as input.
        """
        return x


ACTIVATION_FUNCTIONS = {
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
    "identity": torch.nn.Identity(),
    "linear": LinearActivation(),
}


class MLPTorchModel(torch.nn.Module):
    """
    Multilayer Perceptron (MLP) implemented in PyTorch.
    Builds a fully connected neural network with configurable hidden layers
    and activation functions.

    Args:
        input_shape (int): Number of input features.
        output_shape (int): Number of output features (e.g., classes or regression targets).
        in_nn (List[int], optional): Sizes of each hidden layer. Default = [40, 150].
        activations (List[str], optional): Activation functions for each hidden layer.
                                             Supported: ['relu', 'tanh', 'sigmoid', 'identity', 'linear'].
                                             Default = ['tanh', 'relu'].

    Methods:
        weights_init(m): Initialize Linear layer weights with Xavier uniform init.
        get_weights(): Retrieve current weights of the model.
        set_weights(weights): Set model weights from a given list of tensors.
        forward(x): Forward pass of the model.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_shape).
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        in_nn: List[int] = [40, 150],
        activations: List[str] = ["tanh", "relu"],
    ):
        super().__init__()

        layers: list[torch.nn.Module] = []
        previous_shape = input_shape
        for i in range(len(in_nn)):
            layers.append(torch.nn.Linear(previous_shape, in_nn[i]))

            if activations[i] == "relu":
                layers.append(torch.nn.ReLU())
            elif activations[i] == "tanh":
                layers.append(torch.nn.Tanh())

            previous_shape = in_nn[i]

        layers.append(torch.nn.Linear(previous_shape, output_shape))
        self.layers = torch.nn.Sequential(*layers)

    @staticmethod
    def weights_init(m: torch.nn.Module) -> None:
        """
        Initialize Linear layer weights using Xavier uniform initialization.
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_weights(self) -> List[torch.Tensor]:
        """
        Retrieve the weights of the model.

        Returns:
            List[torch.Tensor]: List of tensors containing the model's parameters.
        """
        weights = []
        for params in self.parameters():
            weights.append(params.data)
        return weights

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """
        Load weights into the model.

        Args:
            weights (List[torch.Tensor]): List of tensors representing weights,
                                            in the same order as returned by `get_weights`.
        """
        for i, param in enumerate(self.parameters()):
            param.data = weights[i].float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_shape).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_shape).
        """
        x = self.layers(x)
        return x


class CNNTorch(torch.nn.Module):
    """
    1D Convolutional Neural Network (CNN) implemented in PyTorch.
    Applies a 1D convolution followed by fully connected layers.

    Args:
        input_shape (int): Number of input features.
        output_shape (int): Number of output units.
        filters (int): Number of convolutional filters.
        in_nn (List[int]): Sizes of hidden layers.
        activations (List[str]): Activation functions for each hidden layer.
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        filters: int,
        in_nn: List[int],
        activations: List[str],
    ):
        super(CNNTorch, self).__init__()

        assert len(activations) == len(in_nn), (
            "Mismatch in number of activations and in_nn layers"
        )

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Convolutional Layer
        self.conv1 = torch.nn.Conv1d(1, filters, kernel_size=1)
        flattened_size = (
            self.conv1(torch.zeros(1, 1, self.input_shape)).view(1, -1).shape[1]
        )

        # Dynamic creation of linear layers followed by activation layers
        layers: list[torch.nn.Module] = []
        in_features = flattened_size
        for i, (out_features, activation) in enumerate(zip(in_nn, activations)):
            layers.append(torch.nn.Linear(in_features, out_features))
            if activation not in ["identity", "linear"]:
                layers.append(ACTIVATION_FUNCTIONS[activation])
            # layers.append(torch.nn.Dropout(p=0.1))  # adding dropout here

            in_features = out_features

        self.layers = torch.nn.Sequential(*layers)

        self.d2 = torch.nn.Dropout(p=0.6)
        # Output Layer
        self.out = torch.nn.Linear(in_nn[-1], self.output_shape)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input tensor of shape (batch_size, input_shape).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_shape).
        """
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x, dtype=torch.float32)
        else:
            x_t = x

        if len(x_t.shape) == 2:
            x_t = x_t.unsqueeze(1)

        x_t = torch.relu(self.conv1(x_t))
        x_t = x_t.view(x_t.size(0), -1)

        x_t = self.layers(x_t)

        x_t = self.d2(x_t)

        x_t = self.out(x_t)

        return x_t


class MLPTorchModelWithDropout(torch.nn.Module):
    """
    Multilayer Perceptron (MLP) implemented in PyTorch.

    This model builds a fully connected feedforward neural network with configurable
    hidden layers and activation functions.

    Args:
        input_shape (int): Number of input features.
        output_shape (int): Number of output units.
        in_nn (List[int], optional): Sizes of hidden layers.
        activations (List[str], optional): Activation functions for each hidden layer.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_shape).
    """

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        in_nn: List[int] = [40, 150],
        activations: List[str] = ["tanh", "relu"],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[torch.nn.Module] = []
        previous_shape = input_shape
        for i in range(len(in_nn)):
            layers.append(torch.nn.Linear(previous_shape, in_nn[i]))

            if activations[i] == "relu":
                layers.append(torch.nn.ReLU())
            elif activations[i] == "tanh":
                layers.append(torch.nn.Tanh())

            # Add dropout after each activation
            if i < 1:
                layers.append(torch.nn.Dropout(p=dropout))

            previous_shape = in_nn[i]

        # Output layer (no activation, no dropout)
        layers.append(torch.nn.Linear(previous_shape, output_shape))
        self.layers = torch.nn.Sequential(*layers)

    @staticmethod
    def weights_init(m: torch.nn.Module) -> None:
        """Initialize model weights using Xavier uniform for Linear layers."""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_weights(self) -> List[torch.Tensor]:
        """
        Retrieve the weights of the model.

        Returns:
            List[torch.Tensor]: List of tensors containing the model's parameters.
        """
        return [params.data for params in self.parameters()]

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        """
        Load weights into the model.

        Args:
            weights (List[torch.Tensor]): List of tensors representing weights,
                                         in the same order as returned by `get_weights`.
        """
        for i, param in enumerate(self.parameters()):
            param.data = weights[i].float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP with Dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_shape).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_shape).
        """
        return self.layers(x)


if __name__ == "__main__":
    nn_model = MLPTorchModel(input_shape=10, output_shape=5)
    print(nn_model)
