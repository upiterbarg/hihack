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


class HierarchicalLSTM(nn.Module):
    def __init__(self, shape, action_space, flags, device, num_strategies=13):
        super(HierarchicalLSTM, self).__init__()

        self.flags = flags
        self.num_actions = len(action_space)
        self.num_strategies = num_strategies

        self.use_prev_action = flags.use_prev_action

        self.topline_encoder = TopLineEncoder()
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        pixel_size = flags.pixel_size
        if flags.crop_dim == 0:
            screen_shape = (24 * pixel_size, 80 * pixel_size)
        else:
            screen_shape = (flags.crop_dim * pixel_size, flags.crop_dim * pixel_size)

        self.screen_encoder = torch.jit.script(ScreenEncoder(screen_shape))

        self.prev_actions_dim = self.num_actions if self.use_prev_action else 0

        self.strategy_dim = self.num_strategies

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )

        self.policy_hidden_dim = 256
        self.strategy_hidden_dim = 128
        self.hidden_dim = 512

        self.strategy_encoder = nn.Linear(self.hidden_dim, self.num_strategies)

        self.core = nn.LSTM(self.h_dim, self.hidden_dim, num_layers=1)

        self.policies = nn.ModuleDict(
            [[f'{i}', nn.Sequential(nn.Linear(self.hidden_dim, self.policy_hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(self.policy_hidden_dim, self.num_actions))] for i in range(self.num_strategies)]
        )

        self.baseline = nn.Linear(self.hidden_dim, 1)
        self.version = 0
        self.action_masks = {}

        self.gumbel_softmax_tau = 1
        if 'gumbel_softmax_tau' in flags:
            self.gumbel_softmax_tau = flags.gumbel_softmax_tau

    def initial_state(self, batch_size=1):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state, last_ttyrec_data=None, return_strategywise_logits=False):
        T, B, C, H, W = inputs["screen_image"].shape

        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        st = [
            self.topline_encoder( topline.float(memory_format=torch.contiguous_format).view(T * B, -1)),
            self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)),
            self.screen_encoder(inputs["screen_image"].float(memory_format=torch.contiguous_format).view(T * B, C, H, W)),
        ]
        if self.use_prev_action:
            st.append(torch.nn.functional.one_hot(inputs["prev_action"], self.num_actions).view(T * B, -1))

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
        strategy_logits = self.strategy_encoder(core_output).view(T * B, -1)

        all_policy_logits = torch.stack([self.policies[str(i)](core_output) for i in range(self.num_strategies)], axis=0)
        strategies = F.gumbel_softmax(strategy_logits, tau=self.gumbel_softmax_tau, hard=True).bool().unsqueeze(-1).expand((-1, -1, all_policy_logits.shape[-1]))
        out_policy_logits = torch.sum(torch.mul(all_policy_logits, torch.swapaxes(strategies, 0, 1)), axis=0).view(T, B, -1)
        out_action = torch.multinomial(F.softmax(out_policy_logits.reshape(T * B, -1), dim=1), num_samples=1).long().view(T, B)


        # -- [B' x 1]
        baseline = self.baseline(core_output)
        baseline = baseline.view(T, B)
        strategy_logits = strategy_logits.view(T, B, -1)

        version = torch.ones_like(out_action) * self.version

        output = dict(
            policy_logits=out_policy_logits,
            all_policy_logits=torch.swapaxes(torch.swapaxes(all_policy_logits, 0, 1), 1, 2),
            baseline=baseline,
            action=out_action,
            version=version,
            strategy_logits=strategy_logits,
        )

        if return_strategywise_logits:
            output['strategywise_policy_logits'] = all_policy_logits

        return (output, core_state)
