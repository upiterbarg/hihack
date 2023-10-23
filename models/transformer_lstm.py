import json
import numpy as np
import os
import pathlib
import pdb
import torch
import sys
import math
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
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

class TransformerLSTM(nn.Module):
    def __init__(self, shape, action_space, flags, device):
        super(TransformerLSTM, self).__init__()

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

        ## second copy of encoders
        self.topline_encoder2 = TopLineEncoder()
        self.bottomline_encoder2 = torch.jit.script(BottomLinesEncoder())
        self.screen_encoder2 = torch.jit.script(ScreenEncoder(screen_shape))
        ###

        self.prev_actions_dim = 128 if self.use_prev_action else 0

        self.h_dim = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )

        self.hidden_dim = 512
        
        self.core = nn.LSTM(self.h_dim, self.hidden_dim, num_layers=1)

        self.num_attention_heads = flags.num_attention_heads
        self.num_transformer_encoder_layers = flags.num_transformer_layers
        self.hidden_dim = self.h_dim + self.hidden_dim
        core_trnsfrmr_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_attention_heads, norm_first=True, activation='gelu')
        self.core_trnsfrmr = nn.TransformerEncoder(core_trnsfrmr_layer, num_layers=self.num_transformer_encoder_layers) # test round 1 uses 4 layers
        self.positional_encoder = PositionalEncoding(self.hidden_dim)
       
        self.policy = nn.Linear(self.hidden_dim, self.num_actions)

        self.baseline = nn.Linear(self.hidden_dim, 1)
        self.version = 0
        self.inference_unroll_length = flags.unroll_length if not 'inference_unroll_length' in flags else flags.inference_unroll_length

        self.wrapped = False

    def initial_state(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, self.inference_unroll_length, self.inference_unroll_length), # transformer portion 0
            torch.rand(self.inference_unroll_length, batch_size, self.hidden_dim), # transformer portion 1
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size), # lstm portion 0
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) # lstm portion 1
            
            
        )

    def get_encodings(self, inputs, for_lstm=False):
        T, B, C, H, W = inputs["screen_image"].shape

        topline = inputs["tty_chars"][..., 0, :]
        bottom_line = inputs["tty_chars"][..., -2:, :]

        if for_lstm or not hasattr(self, 'topline_encoder2'):
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
        else:
            st = [
                self.topline_encoder2(
                    topline.float(memory_format=torch.contiguous_format).view(T * B, -1)
                ),
                self.bottomline_encoder2(
                    bottom_line.float(memory_format=torch.contiguous_format).view(T * B, -1)
                ),
                self.screen_encoder2(
                    inputs["screen_image"]
                    .float(memory_format=torch.contiguous_format)
                    .view(T * B, C, H, W)
                ),
            ]

        if self.use_prev_action:
            st.append(torch.nn.functional.one_hot(inputs["prev_action"], self.prev_actions_dim).view(T * B, -1))

        st = torch.cat(st, dim=1)
        return st



    def forward(self, inputs, core_state=None, last_ttyrec_data=None):
        T, B, C, H, W = inputs["screen_image"].shape
        st_lstm = self.get_encodings(inputs, for_lstm=True)
        st_trnsfrmr = self.get_encodings(inputs, for_lstm=False)

        T_eff = T

        if not last_ttyrec_data is None and self.training:
            last_st_lstm = self.get_encodings(last_ttyrec_data, for_lstm=True)
            last_st_trnsfrmr = self.get_encodings(last_ttyrec_data, for_lstm=False)
            T_eff = T * 2 
            st_lstm = torch.cat([last_st_lstm.reshape(T, B, -1), st_lstm.reshape(T, B, -1)], axis=0).reshape(T_eff * B, -1)
            st_trnsfrmr = torch.cat([last_st_trnsfrmr.reshape(T, B, -1), st_trnsfrmr.reshape(T, B, -1)], axis=0).reshape(T_eff * B, -1)
            self.wrapped = True

        c0, c1, c2, c3 = core_state
        trnsfrmr_core_state = c0, c1
        lstm_core_state = c2, c3

        lstm_core_input = st_lstm.view(T_eff, B, -1)
        lstm_core_output_list = []
        
        if self.wrapped:
            notdone = torch.cat([(~last_ttyrec_data["done"]).float(), (~inputs["done"]).float()], axis=0)
        else:
            notdone = (~inputs["done"]).float()

        notdone_mask = torch.ones((T_eff, T_eff)).repeat(B, 1, 1).to(lstm_core_input.device)

        i = 0
        for input, nd in zip(lstm_core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, hidden_size)
            nd = nd.view(1, -1, 1)
            lstm_core_state = tuple(nd * t for t in lstm_core_state)
            output, lstm_core_state = self.core(input.unsqueeze(0), lstm_core_state)
            lstm_core_output_list.append(output)

            if i < T_eff-1:
                nd = notdone[i].view(-1, 1, 1)
                notdone_mask[:, i+1:, :i+1] *= nd

            i += 1

        lstm_core_output = torch.flatten(torch.cat(lstm_core_output_list), 0, 1)

        st = torch.cat([st_trnsfrmr, lstm_core_output], dim=1)

        trnsfrmr_core_input = st.reshape(T_eff, B, -1)
        if not self.training:
            prev_mask, prev_encodings = trnsfrmr_core_state
            prev_mask = prev_mask.squeeze(0)
            trnsfrmr_core_input = torch.cat([prev_encodings[1:], trnsfrmr_core_input], axis=0)
            trnsfrmr_core_mask = torch.stack(
                [torch.cat([torch.cat([prev_mask[i, 1:, 1:], prev_mask[i, -1, 1:].unsqueeze(0)], axis=0) * notdone[-1, i], torch.zeros((self.inference_unroll_length, 1)).to(trnsfrmr_core_input.device)], axis=1) for i in range(B)]
            )
            trnsfrmr_core_mask[:, -1, -1] = 1
            trnsfrmr_core_state = (trnsfrmr_core_mask.detach().clone().unsqueeze(0), 
                trnsfrmr_core_input.detach().clone()
            )
            for i in range(B):
                trnsfrmr_core_mask[i].fill_diagonal_(1)
            trnsfrmr_core_mask = (trnsfrmr_core_mask.float().masked_fill(trnsfrmr_core_mask == 0, float("-inf")).masked_fill(trnsfrmr_core_mask == 1, float(0.0))).to(device=trnsfrmr_core_input.device)
            trnsfrmr_core_mask = torch.repeat_interleave(trnsfrmr_core_mask, self.num_attention_heads, dim=1).reshape(B * self.num_attention_heads, self.inference_unroll_length, self.inference_unroll_length)
            T = trnsfrmr_core_input.shape[0]
        elif self.wrapped:  
            mask1 = (torch.triu(torch.ones(T_eff, T_eff)) == 1).transpose(0, 1)
            mask2 = F.pad((torch.triu(torch.ones(T, T)) == 1).transpose(0, 1), (0, T, T, 0))
            trnsfrmr_core_mask = mask1.long() + mask2.long()
            trnsfrmr_core_mask[trnsfrmr_core_mask != 1] = 0
            trnsfrmr_core_mask = (trnsfrmr_core_mask.float().masked_fill(trnsfrmr_core_mask == 0, float("-inf")).masked_fill(trnsfrmr_core_mask == 1, float(0.0))).to(device=trnsfrmr_core_input.device)
        else:
            trnsfrmr_core_mask = generate_square_subsequent_mask(T, trnsfrmr_core_input.device)


        trnsfrmr_core_input = self.positional_encoder(trnsfrmr_core_input)
        trnsfrmr_core_output = self.core_trnsfrmr(trnsfrmr_core_input, trnsfrmr_core_mask)
        trnsfrmr_core_output = torch.flatten(trnsfrmr_core_output, 0, 1)

        # -- [B' x A]
        policy_logits = self.policy(trnsfrmr_core_output)

        # -- [B' x 1]
        baseline = self.baseline(trnsfrmr_core_output)

        if self.wrapped:
            policy_logits = policy_logits.view(2*T, B, -1)[-T:].view(T * B, -1)
            baseline = baseline.view(2*T, B, -1)[-T:].view(T * B, -1)

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

        c0, c1 = trnsfrmr_core_state
        c2, c3 = lstm_core_state

        core_state = (c0, c1, c2, c3)

        self.wrapped = False
        return (output, core_state)


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
