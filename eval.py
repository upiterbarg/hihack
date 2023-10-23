import copy
import models.utils
import moolib
import numpy as np
import omegaconf
import os
import pathlib
import pdb
import sys
import torch
import tqdm

from argparse import ArgumentParser

base_path = str(pathlib.Path().resolve())
HIHACK_PATH = os.path.join(base_path[:base_path.find('hihack')], 'hihack')
sys.path.insert(0, os.path.join(HIHACK_PATH, 'dungeonsdata-neurips2022/experiment_code'))
import hackrl.core
import hackrl.environment

from hackrl.core import nest

ENVS = None
os.environ['MOOLIB_ALLOW_FORK'] = 'true'


def run_eval(args, 
             model, 
             model_flags, 
             pbar_idx=0, 
             device='cuda' if torch.cuda.is_available() else 'cpu'):
    params = sum(p.numel() for p in model.parameters())
    print(params)

    global ENVS

    num_batches = args.num_batches
    split = args.split
    nproc = args.nproc

    episodes = args.episodes // (num_batches * split)
    model_flags.batch_size = episodes


    ENVS = moolib.EnvPool(
        lambda: hackrl.environment.create_env(model_flags),
        num_processes=nproc,
        batch_size=episodes,
        num_batches=num_batches,
    )

    episodes_left = (
        torch.ones(
            (
                num_batches,
                episodes,
            )
        )
        .long()
        .to(device)
        * split
    )
    current_reward = torch.zeros(
        (
            num_batches,
            episodes,
        )
    ).to(device)

    returns = []
    results = [None, None]
    grand_pbar = tqdm.tqdm(position=0, leave=True)
    pbar = tqdm.tqdm(
        total=episodes * num_batches * split, position=pbar_idx + 1, leave=True
    )

    action = torch.zeros((num_batches, episodes)).long().to(device)
    hs = [model.initial_state(episodes) for _ in range(num_batches)]
    hs = nest.map(lambda x: x.to(device), hs)

    totals = torch.sum(episodes_left).item()
    subtotals = [torch.sum(episodes_left[i]).item() for i in range(num_batches)]
    while totals > 0:
        grand_pbar.update(1)
        for i in range(num_batches):
            if subtotals[i] == 0:
                continue
            if results[i] is None:
                results[i] = ENVS.step(i, action[i])
            outputs = results[i].result()

            env_outputs = nest.map(lambda t: t.to(model_flags.device, copy=True), outputs)
            env_outputs["prev_action"] = action[i]
            current_reward[i] += env_outputs["reward"]

            done_and_valid = env_outputs["done"].int() * episodes_left[i].bool().int()
            finished = torch.sum(done_and_valid).item()
            totals -= finished
            subtotals[i] -= finished

            for j in np.argwhere(done_and_valid.cpu().numpy()):
                returns.append(current_reward[i][j[0]].item())

            current_reward[i] *= 1 - env_outputs["done"].int()
            episodes_left[i] -= done_and_valid
            if finished:
                pbar.update(finished)

            env_outputs = nest.map(lambda x: x.unsqueeze(0), env_outputs)
            with torch.no_grad():
                outputs, hs[i] = model(env_outputs, hs[i])
            action[i] = outputs["action"].reshape(-1)
            results[i] = ENVS.step(i, action[i])

    return returns

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_batches', type=int, default=2)
    parser.add_argument('--split', type=int, default=4)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('-n', '--episodes', type=int, default=1024, help='total number of games to evaluate on')
    parser.add_argument('--model_name_or_path', type=str, default='all', help='which models to evaluate? "all" --> looks for + runs eval on all pretrained hihack models')
    parser.add_argument('--eval_dir', type=str, default='eval_results', help='name of directory where the eval results should be dumped')

    args = parser.parse_args()
    print('ARGS:', args)
    return args


def main():
    args = parse_args()

    eval_dir = os.path.join(HIHACK_PATH, args.eval_dir)
    os.makedirs(eval_dir, exist_ok=True)

    pt_model_ckpts_default_path = os.path.join(HIHACK_PATH, 'pt_model_ckpts')

    # run eval for all pretrained model checkpoints
    if args.model_name_or_path == 'all':
        assert os.path.exists(pt_model_ckpts_default_path)

        for ckpt_name in os.listdir(pt_model_ckpts_default_path):
            model_path = os.path.join(pt_model_ckpts_default_path, ckpt_name)
            nickname = ckpt_name[:ckpt_name.rfind('.')]
            out_fn = os.path.join(eval_dir, nickname + '.txt')

            model, flags = models.utils.load_pt_model_and_flags(model_path, 'cuda')
            flags = omegaconf.OmegaConf.create(flags)
            model.eval()

            returns = run_eval(args, model, flags)

            with open(out_fn, 'a') as f:
                f.writelines([','.join([str(r) for r in returns]) + '\n'])

        return

    # try to unpack pretrained model checkpoint from alias
    if not '.tar' in args.model_name_or_path:
        model_path = os.path.join(pt_model_ckpts_default_path, args.model_name_or_path + '.tar')
        assert os.path.exists(model_path)
   
    # regular eval from path
    else:
        model_path = args.model_name_or_path

    model, flags = models.utils.load_pt_model_and_flags(model_path, 'cuda')
    flags = omegaconf.OmegaConf.create(flags)
    model.eval()

    returns = run_eval(args, model, flags)

    ckpt_name = model_path if not '/' in model_path else model_path[model_path.rfind('/')+1:]
    nickname = ckpt_name[:ckpt_name.rfind('.')]
    out_fn = os.path.join(eval_dir, nickname + '.txt')

    with open(out_fn, 'a') as f:
        f.writelines([','.join([str(r) for r in returns]) + '\n'])


if __name__ == '__main__':
    main()


