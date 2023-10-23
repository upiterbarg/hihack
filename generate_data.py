import functools
import gym
import multiprocessing
import nle.nethack as nh
import numpy as np
import os
import pathlib
import pdb
import sys
import time

from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm

from autoascend_env_wrapper import AutoAscendEnvWrapper
base_path = str(pathlib.Path().resolve())
HIHACK_PATH = os.path.join(base_path[:base_path.find('hihack')], 'hihack')

def get_seeds(n, 
              target_role, 
              start_seed=0):

    if target_role == 'null':
        return np.array([i for i in range(start_seed, n+start_seed)])

    relevant_seeds = []
    with tqdm(total=n) as pbar:
        while not len(relevant_seeds) == n:
            candidate_seed = start_seed
            while 1:
                env = gym.make('NetHackChallenge-v0')
                env.seed(candidate_seed, candidate_seed)
                obs = env.reset()
                blstats = agent_lib.BLStats(*obs['blstats'])
                character_glyph = obs['glyphs'][blstats.y, blstats.x]
                if any([nh.permonst(nh.glyph_to_mon(character_glyph)).mname.startswith(role) for role in target_role]):
                    break
                candidate_seed += 10**5
                candidate_seed = candidate_seed % sys.maxsize
                env.close()
            if not candidate_seed in relevant_seeds and not candidate_seed in restricted_seeds:
                relevant_seeds += [candidate_seed]
                pbar.update(1)
            start_seed += 1
    return np.array(relevant_seeds).astype(int)

def gen_and_write_episode(idx, 
                          start_i, 
                          total_rollouts, 
                          data_dir, 
                          seeds, 
                          zbase=1):
    with tqdm(total=total_rollouts, position=idx, desc=str(os.getpid())) as pbar:
        for game_id in range(start_i, start_i + total_rollouts):
            # unpack game seed
            if game_id >= seeds.shape[0]:
                break
            game_seed = seeds[game_id]

            env = AutoAscendEnvWrapper(
                gym.make(
                    'NetHackChallenge-v0', 
                    no_progress_timeout=1000, 
                    savedir=os.path.join(data_dir, f'{game_seed}'), 
                    save_ttyrec_every=1, 
                    max_episode_steps=200000000
                ), 
                agent_args=dict(panic_on_errors=True, verbose=False)
            )
            env.env.seed(game_seed, game_seed)
            try:
                env.main()
            except BaseException:
                pass

            pbar.update(1)
    return 1

def create_dataset(args):
    # set main filepath
    data_dir = os.path.join(HIHACK_PATH, args.base_dir, args.dataset_name)
    os.makedirs(data_dir, exist_ok=True)
   
    # first determine n unique seeds 
    if args.role is None:
        role = 'null'
    else:
        role = args.role

    relevant_seeds = get_seeds(args.episodes, role, args.seed)

    seeds_done = [int(f[f.rfind('/')+1:]) for f in os.listdir(data_dir)]
    relevant_seeds = np.array(list(set(list(relevant_seeds)).difference(set(seeds_done))))
    print(f'{relevant_seeds.shape[0]} seeds generated!')


    ## parallelize dataset generation + saving
    num_proc = max(min(multiprocessing.cpu_count(), args.cores), 1) # use no more than the number of available cores
    num_rollouts_per_proc = (relevant_seeds.shape[0] // num_proc) + 1
    gen_helper_fn = functools.partial(
        gen_and_write_episode, 
        data_dir=data_dir, 
        seeds=relevant_seeds, 
        zbase=int(np.log10(args.episodes) + 0.5)
    )

    # generate remaining args
    gen_args = []
    start_i = 0
    for j, proc in enumerate(range(num_proc - 1)):
        gen_args += [[j, start_i, num_rollouts_per_proc]]
        start_i += num_rollouts_per_proc 
    if relevant_seeds.shape[0] - start_i > 0:
        gen_args += [[num_proc - 1, start_i, relevant_seeds.shape[0] - start_i]]
    
    # run pool
    pool = multiprocessing.Pool(num_proc)
    runs = [pool.apply_async(gen_helper_fn, args=gen_args[k]) for k in range(num_proc) if len(gen_args) > k]
    results = [p.get() for p in runs]
    
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_dir', default='data', type=str, help='dir where to store data')
    parser.add_argument('--dataset_name', default='test2', type=str)
    parser.add_argument('--seed', default=0, type=int, help='starting random seed')
    parser.add_argument('-c', '--cores', default=4, type=int, help='cores to employ')
    parser.add_argument('-n', '--episodes', type=int, default=10000)
    parser.add_argument('--role', choices=('arc', 'bar', 'cav', 'hea', 'kni',
                                           'mon', 'pri', 'ran', 'rog', 'sam',
                                           'tou', 'val', 'wiz'),
                        action='append')
    parser.add_argument('--panic-on-errors', default=True, action='store_true')

    args = parser.parse_args()

    print('ARGS:', args)
    return args

def main():
    args = parse_args()
    create_dataset(args)

if __name__ == '__main__':
    main()
