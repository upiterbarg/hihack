import omegaconf
import os
import pathlib
import pdb
import sys
import torch

from .cdgpt5 import CDGPT5
from .cleaved_hierarchical_policy import CleavedHierarchicalPolicy
from .flat_transformer import FlatTransformer
from .hierarchical_lstm import HierarchicalLSTM
from .hierarchical_transformer_lstm import HierarchicalTransformerLSTM
from .transformer_lstm import TransformerLSTM


from nle.env.base import DUNGEON_SHAPE
from omegaconf import OmegaConf



base_path = str(pathlib.Path().resolve())
hihack_path = os.path.join(base_path[:base_path.find('hihack')], 'hihack')
sys.path.insert(0, os.path.join(hihack_path, 'dungeonsdata-neurips2022/experiment_code/hackrl'))
from tasks import ENVS

MODELS = [
    CDGPT5,
	HierarchicalLSTM,
	HierarchicalTransformerLSTM,
	TransformerLSTM,
    FlatTransformer
]

MODELS_LOOKUP = {c.__name__: c for c in MODELS}


def initialize_weights(flags, model):
    def _initialize_weights(layer):
        if hasattr(layer, "bias") and isinstance(
            layer.bias, torch.nn.parameter.Parameter
        ):
            layer.bias.data.fill_(0)

        if flags.initialisation == "orthogonal":
            if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.orthogonal_(layer.weight.data, gain=1.0)
        elif flags.initialisation == "xavier_uniform":
            if type(layer) in [torch.nn.Conv2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=1.0)
            else:
                pass
        else:
            pass

    model.apply(_initialize_weights)

def load_flags(load_path):
    out = torch.load(load_path)
    return omegaconf.OmegaConf.create(out['flags'])


def create_model(flags, device, model_type=None):
    model_type = model_type if not model_type is None else flags.model
    try:
        model_cls = MODELS_LOOKUP[model_type]
    except KeyError:
        raise NotImplementedError("model=%s" % flags.model) from None

    action_space = ENVS[flags.env.name](savedir=None).actions

    model = model_cls(DUNGEON_SHAPE, action_space, flags, device)
    model.to(device=device)

    initialize_weights(flags, model)
    return model


def load_pt_model_and_flags(load_path, device):
    out = torch.load(load_path, map_location=device)

    flags = omegaconf.OmegaConf.create(out['flags'])

    if flags.model == 'CleavedHierarchicalPolicy':
        assert len(out['submodule_flags']) > 0
        submodule_flags = omegaconf.OmegaConf.create(out['submodule_flags'])
        high_level_model = create_model(submodule_flags, device, model_type=flags.high_level_model)
        low_level_model = create_model(flags, device, model_type=flags.low_level_model)
        model = CleavedHierarchicalPolicy(flags, high_level_model, low_level_model)
    else:
        model = create_model(flags, device)

    if 'learner_state' in out:
        model_state_dict = out['learner_state']['model']
    else:
        model_state_dict = out['model']
    
    model.load_state_dict(model_state_dict)

    return model, out['flags']