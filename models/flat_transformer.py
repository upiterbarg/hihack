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

from .transformer_lstm import (
    generate_square_subsequent_mask,
    PositionalEncoding
)

base_path = pathlib.Path().resolve()
sys.path.insert(0, os.path.join(base_path, '..', 'dungeonsdata-neurips2022/experiment_code/hackrl/models'))
from chaotic_dwarf import (
    TopLineEncoder,
    BottomLinesEncoder,
    ScreenEncoder,
    conv_outdim
)


class FlatTransformer(nn.Module):
    def __init__(self, shape, action_space, flags, device):
        super(FlatTransformer, self).__init__()
        
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

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )

        self.num_attention_heads = flags.num_attention_heads 
        self.num_transformer_encoder_layers = flags.num_transformer_layers
        core_layer = nn.TransformerEncoderLayer(d_model=self.h_dim, nhead=self.num_attention_heads)
        self.core = nn.TransformerEncoder(core_layer, num_layers=self.num_transformer_encoder_layers)
        self.positional_encoder = PositionalEncoding(self.h_dim)

        self.policy_hidden_dim = 1024
        self.policy = nn.Sequential(nn.Linear(self.h_dim, self.policy_hidden_dim),
            nn.ELU(),
            nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
            nn.ELU(),
            nn.Linear(self.policy_hidden_dim, self.num_actions)
        )
        self.baseline = nn.Linear(self.h_dim, 1)

        self.version = 0
        self.inference_unroll_length = 1

    def initial_state(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, self.inference_unroll_length, self.inference_unroll_length),
            torch.rand(self.inference_unroll_length, batch_size, self.h_dim)
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

        core_input = st.reshape(T, B, -1)
        notdone = (~inputs["done"]).float()
        if not self.training:
            prev_mask, prev_encodings = core_state
            prev_mask = prev_mask.squeeze(0)
            core_input = torch.cat([prev_encodings[1:], core_input], axis=0)
            core_mask = torch.stack(
                [torch.cat([torch.cat([prev_mask[i, 1:, 1:], prev_mask[i, -1, 1:].unsqueeze(0)], axis=0) * notdone[-1, i], torch.zeros((self.inference_unroll_length, 1)).to(core_input.device)], axis=1) for i in range(B)]
            )
            core_mask[:, -1, -1] = 1
            core_state = (core_mask.detach().clone().unsqueeze(0), 
                core_input.detach().clone()
            )
            for i in range(B):
                core_mask[i].fill_diagonal_(1)
            core_mask = (core_mask.float().masked_fill(core_mask == 0, float("-inf")).masked_fill(core_mask == 1, float(0.0))).to(device=core_input.device)

            core_mask = torch.repeat_interleave(core_mask, self.num_attention_heads, dim=1).reshape(B * self.num_attention_heads, self.inference_unroll_length, self.inference_unroll_length)
            T = core_input.shape[0]
        else:
            core_mask = generate_square_subsequent_mask(T, core_input.device)

        core_input = self.positional_encoder(core_input)
        core_output = self.core(core_input, core_mask)
        core_output = torch.flatten(core_output, 0, 1)

        # -- [B' x A]
        policy_logits = self.policy(core_output)

        # -- [B' x 1]
        baseline = self.baseline(core_output)

        action = torch.multinomial(F.softmax(policy_logits + 1e-5, dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        version = torch.ones_like(action) * self.version


        if not self.training:
            action = action[-1].unsqueeze(0)
            baseline = baseline[-1].unsqueeze(0)
            policy_logits = policy_logits[-1].unsqueeze(0)
            version = version[-1].unsqueeze(0)

        output = dict(
            policy_logits=policy_logits,
            baseline=baseline,
            action=action,
            version=version,
        )
        
        return (output, core_state)
