import numpy as np
import copy
from typing import Dict, List
from utils import *


class FRPAgent(object):

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_state: int,
        S: List,
        Pi: List,
        FRs: List,
        max_sweeps: int = 5) -> None:
        """An agent which implements FR-Planning.

        Args:
            number_of_states (int)
            number_of_actions (int)
            initial_state (int)
            S (List): list of states 
            Pi (List): list of N base policies, |S| x |A| matrices
            FRs (List): associated |S| x |S| FR matrices 
            max_sweeps (int, optional): Number of allowed policies. Defaults to 5.
        """

        self._number_of_actions = number_of_actions
        self._number_of_states = number_of_states
        self._state = initial_state
        self._initial_state = initial_state 
        self._S = S # state space

        self.F = np.stack(FRs) # N x |S| x |S|
        self.pi_F, self.s_F = self.dynamic_plan(initial_state, max_sweeps=max_sweeps) # plan for each possible goal state
        self.Pi = Pi
        self.r = np.zeros(number_of_states)
        self._max_sweeps = max_sweeps


    @property
    def Policies(self):
        return self.pi_F
    
    @property
    def Subgoals(self):
        return self.s_F
        

    def dynamic_plan(
        self,
        initial_state: int,
        max_sweeps: int = 5) -> List[Dict]:
        """run FRP

        Args:
            initial_state (int): index of start state
            max_sweeps (int, optional): maximum number of sweeps. defaults to 5.

        Returns:
            [Dict, Dict]: return pi^F, s^F (dicts mapping goal state to corresponding pi^F, s^F)
        """
        

        N, nS, _ = self.F.shape

        pi_F_dict = {} # a policy in each state, for each state as a possible subgoal
        s_F_dict = {} # same as above but for subgoals
        for goal_state in self._S: 

            # reshape F to be |S| x N x |S|, s.t. F(n, s, s') -> F_reshaped(s, n, s')
            F = copy.copy(self.F)
            F_reshaped = np.transpose(F, axes=(1, 0, 2)) 

            # initialize values, subgoals, policies
            Gamma_prev = -np.inf * np.ones(nS)
            pi_F = np.argmax(F[:, :, goal_state], axis=0) # which policy (by index) in each state 
            Gamma = np.max(F[:, :, goal_state], axis=0) # |S|-vector
            s_F = goal_state * np.ones(nS, dtype=int)


            # iteratively refine values
            m = 0 # m is the number of switch states allowed on a trajectory 
            while m < max_sweeps:
                
                Gamma_prev = Gamma
                for s in range(nS): 
                    F_s = F_reshaped[s, ...] # N x nS (s')
                    if s != goal_state: F_s[:, s] = 0; 
                    FG = F_s * np.tile(Gamma[None, :], [N, 1]) 
                    Gamma[s] = np.max(FG)
                    piF, sF = np.unravel_index(np.argmax(FG.flatten()), shape=[N, nS])
                    pi_F[s], s_F[s] = piF, int(sF)

                m += 1

            pi_F_dict[goal_state] = pi_F
            s_F_dict[goal_state] = s_F


        return pi_F_dict, s_F_dict


    def construct_plan(self, initial_state: int, goal_state: int, max_iters: int = 100) -> List:
        """construct an explicit plan using pi^F, s^F

        Args:
            initial_state (int): start state
            goal_state (int): goal state
            max_iters (int, optional): max # of steps in the plan. defaults to 100.

        Returns:
            List: list of policies and their associated subgoals
        """
        Lambda = []
        s, m = initial_state, 0
        while s != goal_state:
            Lambda.append((self.pi_F[goal_state][s], self.s_F[goal_state][s]))
            s = self.s_F[goal_state][s]
            m += 1
            if m >= max_iters: break; 

        return Lambda


    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step using the planning output. 

        Args:
            reward (float): Last step reward. Ignored. 
            discount (float): Discount factor. Ignored.
            next_state (int)

        Returns:
            int: The action, an index from [0, 1,..., |A|-1]
        """

        
        self._current_pi = self.Pi[self.pi_F[np.argmax(self.r)][next_state]]

        # act greedily wrt current policy
        next_action = greedy(self._current_pi.q[next_state, :])

        self._state = next_state

        return next_action

