import json
import numpy as np
import os
import pathlib
import pdb
import sys
import torch

from nle import nethack
from nle.nethack.actions import ACTIONS as A
from torch import nn
from torch.nn import functional as F


base_path = pathlib.Path().resolve()
sys.path.insert(0, os.path.join(base_path, '..', 'dungeonsdata-neurips2022/experiment_code/hackrl/models'))
from chaotic_dwarf import (
    TopLineEncoder,
    BottomLinesEncoder,
    ScreenEncoder,
    conv_outdim
)


class CDGPT5(nn.Module):
    def __init__(self, shape, action_space, flags, device):
        super(CDGPT5, self).__init__()

        self.flags = flags
        self.num_actions = len(action_space)
        self.use_prev_action = flags.use_prev_action

        self.topline_encoder = TopLineEncoder()
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        pixel_size = flags.pixel_size
        if flags.crop_dim == 0:
            screen_shape = (24 * pixel_size, 80 * pixel_size)
        else:
            screen_shape = (flags.crop_dim * pixel_size, flags.crop_dim * pixel_size)

        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.prev_actions_dim = 128 if self.use_prev_action else 0
        self.hidden_dim = 512

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )
  
        self.core = nn.LSTM(self.h_dim, self.hidden_dim, num_layers=1)

        if flags.cdgpt5_xxl_decoder:
            self.policy_hidden_dim = 1024
            self.policy = nn.Sequential(nn.Linear(self.hidden_dim, self.policy_hidden_dim),
                nn.ELU(),
                nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
                nn.ELU(),
                nn.Linear(self.policy_hidden_dim, self.num_actions)
            )
        else:
            self.policy = nn.Linear(self.hidden_dim, self.num_actions)

        self.baseline = nn.Linear(self.hidden_dim, 1)
        self.version = 0
        self.inference_unroll_length = flags.unroll_length if not 'inference_unroll_length' in flags else flags.inference_unroll_length

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=None):
        T, B, C, H, W = inputs["screen_image"].shape

        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        st = [
            self.topline_encoder(
                topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.bottomline_encoder(
                bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
            ),
            self.screen_encoder(
                inputs["screen_image"]
                .float(memory_format=torch.contiguous_format)
                .view(T * B, C, H, W)
            ),
        ]
        if self.use_prev_action:
            st.append(torch.nn.functional.one_hot(inputs["prev_action"], self.prev_actions_dim).view(T * B, -1))

        st = torch.cat(st, dim=1)

        core_input = st.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()

        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * t for t in core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        # -- [B' x A]
        policy_logits = self.policy(core_output)

        # -- [B' x 1]
        baseline = self.baseline(core_output)

        action = torch.multinomial(F.softmax(policy_logits + 1e-5, dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        version = torch.ones_like(action) * self.version

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            version=version,
        )

        return (output, core_state)
