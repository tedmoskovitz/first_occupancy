import numpy as np
from typing import Dict
from utils import * 



def run_experiment_episodic(
    env,
    agent,
    number_of_episodes: int,
    eval_only: bool = False,
    max_ep_len: int = 20,
    display_eps: int = None,
    respect_done: bool = True
    ) -> Dict:
    """
    run an experiment
    """
    return_hist = []
    deltas = []
    trajectories = []
    if hasattr(agent, '_eval'): agent._eval = eval_only; 
    if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
    if hasattr(agent, 'reset'): agent.reset();

    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(1, number_of_episodes+1):

        if hasattr(agent, 'reset'): agent.reset();
        reward, discount, next_state, done = env.reset()
        if respect_done: agent._state = next_state; 
        if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
        elif hasattr(agent, 'w') and hasattr(env, 'r'): agent.w = env.r; 
        action = agent.step(reward, discount, next_state)
        z = reward
        traj = [env.obs_to_state_coords(next_state)]

        for t in range(1, max_ep_len+1):
            
             # effect of action in env
            reward, discount, next_state, done = env.step(action)
            if hasattr(agent, 'r') and hasattr(env, 'r'): agent.r = env.r; 
            elif hasattr(agent, 'w') and hasattr(env, 'r'): agent.w = env.r; 
            # agent takes next step
            action = agent.step(reward, discount, next_state)
            z += (discount ** t) * reward
            traj.append(env.obs_to_state_coords(next_state))
            if done and respect_done: break; 

        return_hist.append(z)

        # display progress 
        if display_eps is not None and i % display_eps == 0:
            flush_print(f"ep {i}/{number_of_episodes}: mean return = {np.mean(return_hist)}")

        trajectories.append(traj)


    results = {
      "return hist": return_hist,
      "trajectory": traj,
      "trajectories": trajectories,
      "deltas": deltas
    }
    if hasattr(agent, "state_values"): results["state values"] = agent.state_values; 
    if hasattr(agent, "q_values"): results["q values"] = agent.q_values; 
    if hasattr(agent, "SR"): results['SR'] = agent.SR; 
    if hasattr(agent, "FR"): results['FR'] = agent.FR; 


    return results 


def run_experiment(env, agent, number_of_steps, display_steps=None, reset_steps=None):
    """
    run an experiment
    """
    mean_reward = 0.0
    reward_hist = []

    # effect of action in env
    reward, discount, next_state, _ = env.reset()
    # agent takes next step
    action = agent.step(reward, discount, next_state)

    max_state = next_state # riverswim
    max_r = reward # sixarms

    for i in range(1, number_of_steps+1):

        # effect of action in env
        if reset_steps is not None and i % reset_steps == 0:
            reward, discount, next_state, _ = env.reset()
        else: reward, discount, next_state, _ = env.step(action);
        # agent takes next step
        action = agent.step(reward, discount, next_state)

        mean_reward = reward / i + (1 - 1/i) * mean_reward
        reward_hist.append(reward)

        # display progress 
        if display_steps is not None and i % display_steps == 0:
            flush_print(f"step {i}/{number_of_steps}: total reward = {sum(reward_hist)}")#, max state = {max_state}")

    results = {
      'reward_hist': reward_hist
    }
    if hasattr(agent, "state_values"): results["state values"] = agent.state_values; 
    if hasattr(agent, "q_values"): results["q values"] = agent.q_values; 
    if hasattr(agent, "return_est"):  results["return"] = agent.return_est; 


    return results 

