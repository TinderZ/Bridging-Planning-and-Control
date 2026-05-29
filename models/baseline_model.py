# baseline_model.py
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Tuple, List, Optional

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict

from models.common_layers import build_conv_layers, build_fc_layers
# from models.actor_critic_lstm import ActorCriticLSTM


# Helper nn.Module layers for preprocessing (Keep these as they are)
class PermuteChannels(nn.Module):
    """Permutes tensor dimensions from (B, H, W, C) to (B, C, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)

class NormalizeUint8(nn.Module):
    """Normalizes a uint8 tensor (0-255) to a float tensor (0.0-1.0)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0


class BaselineModel(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Box,  # Box类型，只包含RGB图像
        action_space: gym.spaces.Discrete,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        # --- Custom args ---
        conv_filters: List[List],
        fcnet_hiddens: List[int],
        lstm_hidden_size: int = 128,
        conv_activation: str = "relu",
        fc_activation: str = "relu",
        use_orthogonal_init: bool = True
    ):
        nn.Module.__init__(self)
        RecurrentNetwork.__init__(self, obs_space, action_space, lstm_hidden_size, model_config, name)

        self.conv_filters = conv_filters
        self.fcnet_hiddens = fcnet_hiddens
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_activation = conv_activation
        self.fc_activation = fc_activation
        self.use_orthogonal_init = use_orthogonal_init
        self.num_actions = action_space.n

        # --- Encoder ---
        # 观察空间是Box格式，直接包含RGB图像
        h, w, c = obs_space.shape
        print(f"BaselineModel initialized with Box observation space, image shape: {(h, w, c)}")

        self.preprocessor = nn.Sequential(PermuteChannels(), NormalizeUint8())
        self.conv_layers, self.conv_out_dim = build_conv_layers(
            input_shape=(c, h, w), 
            conv_filters=self.conv_filters, 
            activation=self.conv_activation
        )
        self.fc_layers = build_fc_layers(
            input_dim=self.conv_out_dim, 
            fcnet_hiddens=self.fcnet_hiddens, 
            activation=self.fc_activation
        )
        self.encoder_out_dim = self.fcnet_hiddens[-1]

        # --- LSTM Core ---
        self.lstm = nn.LSTMCell(self.encoder_out_dim, self.lstm_hidden_size)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.lstm.weight_ih)
            nn.init.orthogonal_(self.lstm.weight_hh)
            nn.init.constant_(self.lstm.bias_ih, 0.0)
            nn.init.constant_(self.lstm.bias_hh, 0.0)

        # --- Actor Head ---
        self.actor_head = nn.Linear(self.lstm_hidden_size, self.num_actions)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.constant_(self.actor_head.bias, 0.0)

        # --- Critic Head ---
        self.critic_head = nn.Linear(self.lstm_hidden_size, 1)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.constant_(self.critic_head.bias, 0.0)

        # --- Value Storage ---
        self._last_value: Optional[torch.Tensor] = None
        self._last_policy_logits: Optional[torch.Tensor] = None

    def get_initial_state(self) -> List[torch.Tensor]:
        # (保持不变)
        try:
             device = next(self.parameters()).device
        except StopIteration:
             device = torch.device("cpu")
        return [
            torch.zeros(self.lstm_hidden_size, device=device),
            torch.zeros(self.lstm_hidden_size, device=device)
        ]

    @property
    def max_seq_len(self):
         # (保持不变)
        return self.model_config.get("max_seq_len", 20)


    def forward_rnn(self, inputs: torch.Tensor, state: List[torch.Tensor], seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = state[0].shape[0]
        T = -1
        # --- Input shape handling (保持不变) ---
        if inputs.shape[0] == B: # Handle T=1 case during init
            T = 1
            inputs = inputs.unsqueeze(1) # (B, F) -> (B, 1, F)
        elif inputs.shape[0] % B == 0:
            T = inputs.shape[0] // B
            inputs = inputs.reshape(B, T, inputs.shape[-1]) # (B*T, F) -> (B, T, F)
        else:
             # Handle potential errors if shape is unexpected
             raise ValueError(f"Cannot infer time dimension T from input shape {inputs.shape} and batch size {B}")


        outputs = []
        h, c = state
        for t in range(T):
            lstm_input_t = inputs[:, t, :]
            h, c = self.lstm(lstm_input_t, (h, c))
            outputs.append(h)

        output_tensor = torch.stack(outputs, dim=1) # Shape: (B, T, hidden_size)

        # --- *** 计算并存储所有时间步的 Value *** ---
        # Apply critic head to features of all time steps
        vf_out = self.critic_head(output_tensor) # Shape: (B, T, 1)
        self._last_value = vf_out.reshape(-1, 1) # Reshape to (B*T, 1) for storage/value_function

        # 返回 (B*T, hidden_size) 和最终状态 [h, c]
        return output_tensor.reshape(-1, self.lstm_hidden_size), [h, c]


    def forward(
        self,
        input_dict: ModelInputDict,
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播，处理RGB图像观察。
        """
        # 直接从input_dict获取RGB图像观察
        image = input_dict["obs"]

        # 处理图像观察
        processed_obs = self.preprocessor(image.float())
        conv_out = self.conv_layers(processed_obs)
        encoded_features = self.fc_layers(conv_out)

        # 确保状态在正确的设备上
        if state and encoded_features.device != state[0].device:
            state = [s.to(encoded_features.device) for s in state]

        # LSTM处理
        output_features, new_state = self.forward_rnn(encoded_features, state, seq_lens)
        
        # 计算策略logits和价值函数
        logits = self.actor_head(output_features)
        value = self.critic_head(output_features)

        # 存储最新的logits和价值函数估计
        self._last_policy_logits = logits
        self._last_value = value

        return logits, new_state

    def value_function(self) -> torch.Tensor:
        """返回最近一次前向传播计算的价值函数估计"""
        assert self._last_value is not None, "必须先调用forward()"
        return self._last_value.squeeze(-1)

    def policy_logits(self) -> Optional[torch.Tensor]:
        """返回最近一次前向传播计算的策略logits"""
        return self._last_policy_logits
