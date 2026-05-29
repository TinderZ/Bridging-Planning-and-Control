# common_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Optional

# Function to get activation function from string
def get_activation_fn(activation_str: str) -> nn.Module:
    """Maps activation string name to a torch.nn activation module."""
    if activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "tanh":
        return nn.Tanh()
    elif activation_str == "elu":
        return nn.ELU()
    elif activation_str == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_str is None or activation_str == "linear":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation_str}")

# Normc initialization (often used in older RL papers)
# def normc_initializer(std: float = 1.0):
#     """Returns a function to initialize weights with unit norm columns."""
#     def initializer(tensor):
#         out = np.random.randn(*tensor.shape).astype(np.float32)
#         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#         tensor.data.copy_(torch.from_numpy(out))
#     return initializer

def build_fc_layers(
    input_dim: int,
    fcnet_hiddens: List[int],
    activation: str = "relu",
    output_activation: Optional[str] = None,
    layer_norm: bool = False,
    init_std: float = 1.0
) -> nn.Sequential:
    """
    Builds a sequence of fully-connected (dense) layers.

    Args:
        input_dim: Dimension of the input features.
        fcnet_hiddens: List of integers specifying the number of units in each hidden layer.
        activation: Activation function for hidden layers.
        output_activation: Activation function for the output layer (if None, uses linear).
        layer_norm: Whether to add LayerNorm after each hidden activation.
        init_std: Standard deviation for normc initialization.

    Returns:
        A torch.nn.Sequential module representing the FC layers.
    """
    layers = []
    prev_layer_size = input_dim
    act_fn = get_activation_fn(activation)

    for i, size in enumerate(fcnet_hiddens):
        linear_layer = nn.Linear(prev_layer_size, size)
        # Apply normc initialization
        # normc_initializer(init_std)(linear_layer.weight)
        # nn.init.constant_(linear_layer.bias, 0.0)
        # let PyTorch use its default initialization for nn.Linear layers. 
        # PyTorch's default (Kaiming uniform) is a strong modern baseline.

        layers.append(linear_layer)
        layers.append(act_fn)
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        prev_layer_size = size

    # Add output activation if specified
    if output_activation is not None:
        out_act_fn = get_activation_fn(output_activation)
        layers.append(out_act_fn) # Usually applied after the final linear layer if needed, handled outside

    return nn.Sequential(*layers)


def build_conv_layers(
    input_shape: Tuple[int, int, int], # (C, H, W)
    conv_filters: List[List[Union[int, List[int], int]]], # [[out_channels, kernel, stride], ...]
    activation: str = "relu",
    init_std: float = 1.0
) -> Tuple[nn.Sequential, int]:
    """
    Builds a sequence of convolutional layers followed by a Flatten layer.

    Args:
        input_shape: Shape of the input tensor (Channels, Height, Width).
        conv_filters: List describing convolutional layers. Each sublist is
                      [output_channels, kernel_size, stride]. Kernel_size can be
                      an int or a tuple (kH, kW).
        activation: Activation function string.
        init_std: Standard deviation for normc initialization.

    Returns:
        A tuple containing:
            - A torch.nn.Sequential module for the Conv layers + Flatten.
            - The output dimension after flattening.
    """
    layers = []
    c, h, w = input_shape
    act_fn = get_activation_fn(activation)
    prev_channels = c

    for i, (out_channels, kernel, stride) in enumerate(conv_filters):
        # Keras 'valid' padding = PyTorch padding=0
        # Keras 'same' padding requires calculating padding size in PyTorch
        # Assuming 'valid' padding based on old code's Conv2D call structure
        padding = 0 # Keras 'valid' padding default

        # Adapt kernel/stride if they are single ints
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)

        conv_layer = nn.Conv2d(prev_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        # Apply normc initialization (adapted for Conv2d)
        # Standard PyTorch initialization might be preferred (e.g., Kaiming)
        # normc_initializer(init_std)(conv_layer.weight)
        nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity=activation if activation != 'linear' else 'relu') # Kaiming often better for Conv
        nn.init.constant_(conv_layer.bias, 0.0)

        layers.append(conv_layer)
        layers.append(act_fn)
        prev_channels = out_channels

        # Calculate output shape after this layer
        h = (h - kernel[0] + 2 * padding) // stride[0] + 1
        w = (w - kernel[1] + 2 * padding) // stride[1] + 1

    layers.append(nn.Flatten())
    output_dim = prev_channels * h * w

    return nn.Sequential(*layers), output_dim