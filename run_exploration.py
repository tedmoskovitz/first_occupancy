from ast import parse
import numpy as np 
import random
from argparse import ArgumentParser
from tqdm import tqdm
from envs import RiverSwim
from agents import SARSA
from runners import run_experiment
from utils import *




def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='riverswim')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--use_fr', action='store_true', default=False)
    parser.add_argument('--use_sr', action='store_true', default=False)
    parser.add_argument('--gamma_sfr', type=float, default=0.99)
    parser.add_argument('--step_size', type=float, default=0.25)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=1.0) 
    parser.add_argument('--eta', type=float, default=0.05)

    args = parser.parse_args()
    params = vars(args)

    env_dict = {
        'riverswim': RiverSwim 
    }

    rewards = []
    
    for i_exp in tqdm(range(1, params['n_runs']+1)):
        #flush_print(f"training {i_exp}/{params['n_runs']}")

        np.random.seed(params['seed'] + i_exp)
        random.seed(params['seed'] + i_exp)

        # create env
        env = env_dict[params['env']]()

        # create agent
        agent = SARSA(
            env.size,
            2,
            env.get_obs(),
            use_sr=params['use_sr'],
            use_fr=params['use_fr'],
            step_size=params['step_size'],
            epsilon=params['epsilon'],
            eta=params['eta'],
            gamma_sfr=params['gamma_sfr'],
            beta=params['beta'],
            norm=L1
            )

        # run 
        results = run_experiment(
                env, agent, params['total_steps'], display_steps=None
            )

        rewards.append(np.cumsum(results['reward_hist']))

    rewards = np.stack(rewards)

    if params['save']:
        np.savetxt("rewards.csv", rewards, delimiter=',')

    print (f"\nmean total reward: {np.mean(rewards[:, -1])}")



if __name__ == '__main__':
    main()



