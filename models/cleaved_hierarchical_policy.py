import torch

from .hierarchical_transformer_lstm import HierarchicalTransformerLSTM
from torch import nn
from torch.nn import functional as F

class CleavedHierarchicalPolicy(nn.Module):
    def __init__(self, 
                 flags,
                 high_level_model, 
                 low_level_model):
        super(CleavedHierarchicalPolicy, self).__init__()
        self.high_level_model = high_level_model
        self.low_level_model = low_level_model
        self.num_strategies = self.high_level_model.num_strategies

        self.gumbel_softmax_tau = 1
        if 'gumbel_softmax_tau' in flags:
            self.gumbel_softmax_tau = flags.gumbel_softmax_tau

        self.disable_high_level_policy_gradients = flags.disable_high_level_policy_gradients
        self.disable_low_level_policy_gradients = flags.disable_low_level_policy_gradients
        self.version = 0
        self.eps_greedy = flags.eps_greedy if 'eps_greedy' in flags else 1


    def initial_state(self, batch_size=1):
        high_level_core_state = self.high_level_model.initial_state(batch_size)
        low_level_core_state = self.low_level_model.initial_state(batch_size)
        return high_level_core_state + low_level_core_state

    def parameters(self):
        if self.disable_high_level_policy_gradients:
            return self.low_level_model.parameters()
        elif self.disable_low_level_policy_gradients:
            return self.high_level_model.parameters()
        return list(self.low_level_model.parameters()) + list(self.high_level_model.parameters())

    def buffers(self):
        if self.disable_high_level_policy_gradients:
            return self.low_level_model.buffers()
        elif self.disable_low_level_policy_gradients:
            return self.high_level_model.buffers()
        return list(self.low_level_model.buffers()) + list(self.high_level_model.buffers())

    def forward(self, inputs, core_state, last_ttyrec_data=None):
        high_level_core_state, low_level_core_state = core_state[:2], core_state[2:]

        if not last_ttyrec_data is None:
            low_level_out, low_level_core_state = self.low_level_model(inputs, low_level_core_state, return_strategywise_logits=True, last_ttyrec_data=last_ttyrec_data)
        else:
            low_level_out, low_level_core_state = self.low_level_model(inputs, low_level_core_state, return_strategywise_logits=True)
        high_level_out, high_level_core_state = self.high_level_model(inputs, high_level_core_state)

        policy_logits = low_level_out['strategywise_policy_logits']
        strategy_logits = high_level_out['strategy_logits']

        if isinstance(self.low_level_model, HierarchicalTransformerLSTM):
            strategy_logits = torch.cat([strategy_logits[..., -1].unsqueeze(-1), strategy_logits[..., :-1]], axis=-1)

        T, B, _ = strategy_logits.shape

        sample = True

        if self.eps_greedy < 1:
            sample = bool(np.random.binomial(1, self.eps_greedy))

        if sample:
            strategies = F.gumbel_softmax(strategy_logits.reshape(T * B, -1), tau=self.gumbel_softmax_tau, hard=True).bool().unsqueeze(-1).expand((-1, -1, policy_logits.shape[-1]))
            sdim = strategy_logits.size(-1)
            out_policy_logits = torch.sum(torch.mul(policy_logits[:sdim], torch.swapaxes(strategies, 0, 1)), axis=0).view(T, B, -1)
        else:
            strategies = torch.argmax(strategy_logits.reshape(T * B, -1), axis=-1)
            out_policy_logits = policy_logits[strategies, torch.arange(strategies.size(0))].view(T, B, -1)


        out_action = torch.multinomial(F.softmax(out_policy_logits.reshape(T * B, -1), dim=1), num_samples=1).long().view(T, B)

        version = torch.ones_like(out_action) * self.version

        if self.disable_high_level_policy_gradients:
            baseline = low_level_out['baseline']
        else:
            baseline = high_level_out['baseline']

        output = dict(
            policy_logits=out_policy_logits,
            baseline=baseline,
            action=out_action,
            version=version,
            strategy_logits=strategy_logits.view(T, B, -1),
            all_policy_logits=torch.swapaxes(torch.swapaxes(policy_logits, 0, 1), 1, 2),
        )

        core_state =  high_level_core_state + low_level_core_state
        return (output, core_state)