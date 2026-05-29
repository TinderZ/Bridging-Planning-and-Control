# actor_critic_lstm.py
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

#from common_layers import normc_initializer # Assuming normc is desired

class ActorCriticLSTM(nn.Module):
    """
    An LSTM network with separate heads for policy (actor) and value (critic).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_actions: int,
        use_orthogonal_init: bool = True # Common modern practice
    ):
        """
        Args:
            input_size: Dimension of the input features.
            hidden_size: Size of the LSTM hidden state and cell state.
            num_actions: Dimension of the policy output (action logits).
            use_orthogonal_init: Whether to use orthogonal initialization for LSTM and Linear layers.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.actor_head = nn.Linear(hidden_size, num_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

        # Initialization
        if use_orthogonal_init:
            # Orthogonal initialization is often preferred for stability in RL
            nn.init.orthogonal_(self.lstm.weight_ih)
            nn.init.orthogonal_(self.lstm.weight_hh)
            nn.init.constant_(self.lstm.bias_ih, 0.0)
            nn.init.constant_(self.lstm.bias_hh, 0.0)
            # Scale actor output layer init, value layer init to 1.0
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.constant_(self.actor_head.bias, 0.0)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.constant_(self.critic_head.bias, 0.0)
        # else:
        #     # Fallback to normc or default PyTorch init if needed
        #     # Example using normc for critic head (as in original repo)
        #     normc_initializer(0.01)(self.critic_head.weight)
        #     nn.init.constant_(self.critic_head.bias, 0.0)
        #     # Default init for actor and LSTM or apply normc too


    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM and heads.

        Args:
            x: Input tensor of shape (batch_size, input_size).
            state: Tuple containing the hidden state (h) and cell state (c),
                   each of shape (batch_size, hidden_size).

        Returns:
            A tuple containing:
                - Policy logits (batch_size, num_actions).
                - Value function estimate (batch_size, 1).
                - New LSTM state (h_new, c_new).
        """
        h_old, c_old = state
        h_new, c_new = self.lstm(x, (h_old, c_old))

        logits = self.actor_head(h_new)
        value = self.critic_head(h_new)

        return logits, value, (h_new, c_new)

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns zeroed initial states for the LSTM."""
        # device = self.lstm.weight_ih.device # Get device from model parameters
        # h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        # c0 = torch.zeros(batch_size, self.hidden_size, device=device)

        # If device is unknown initially, create on CPU and move later
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros(batch_size, self.hidden_size)
        return h0, c0