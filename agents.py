import numpy as np
from typing import Callable, List
import copy
from utils import * 


class Random(object):

  def __init__(self, number_of_states: int, number_of_actions: int, initial_state: int) -> None:
    """
    A random agent 
    """
    self._number_of_actions = number_of_actions

  def step(self, reward: float, discount: float, next_state: int) -> int:
    next_action = np.random.randint(self._number_of_actions)
    return next_action


class SimpleAgent(object):

    def __init__(self, number_of_states: int, initial_state: int, action_idx: int, n_actions: int = 4, rebound: bool = False) -> None:
        """An agent which just repeats a given action. May take a random action if it hits a wall. 

        Args:
            number_of_states (int): number of states in the environment
            initial_state (int): start state index
            action_idx (int): the index of the action to follow
            n_actions (int, optional): number of possible actions. defaults to 4.
            rebound (bool, optional): whether or not to take a random action if hitting a wall. defaults to False.
        """
        self._state = initial_state
        self._rebound = rebound
        self.q = np.zeros([number_of_states, n_actions])
        
        for i in range(number_of_states):
            self.q[i, :] = np.eye(n_actions)[action_idx]

    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step."""

        if self._rebound and next_state == self._state: # "rebound" if stuck
            action = np.random.choice(self.q.shape[1])
        else:
            action = np.argmax(self.q[next_state, :])

        self._state = next_state

        return action


class GPIAgent(object):

    def __init__(
        self,
        number_of_states: int, 
        number_of_actions: int,
        initial_state: int,
        w: np.ndarray,
        Psi: List) -> None:
        """An agent which performs GPI over a fixed set of policies."""

        self._number_of_actions = number_of_actions
        self._number_of_states = number_of_states
        self._state = initial_state
        self._initial_state = initial_state 
        assert len(w) == number_of_states, "w must be same dimensionality as S"
        self.w = w # should be |S|-dim in tabular case
        # Psi should be a list of n SRs, each is |S| x |A| x |S| (map (s, a) -> expected state occs)
        self.Psi = np.stack(Psi) # n x |S| x |A| x |S|

    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step using GPI."""

        # perform GPE
        Q = self.Psi[:, next_state, :, :] @ self.w # (n x |A| x |S|) @ |S| -> n x |A| 
        # perform GPI
        next_action = np.argmax(np.max(Q, axis=0)) # max over policies, argmax over actions

        self._state = next_state

        return next_action


class VI(object):

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_state: int,
        P: np.ndarray,
        r: np.ndarray,
        gamma: float,
        max_iters: int = 500) -> None:
        """Value Iteration

        Args:
            number_of_states (int): number of states in the environment
            number_of_actions (int): number of available actions
            initial_state (int): starting state index
            P (np.ndarray): |S| x |A| x |S| transition matrix 
            r (np.ndarray): reward vector 
            gamma (float): discount factor
            max_iters (int, optional): max iterations to run. defaults to 500.
        """
        self._q = np.zeros([number_of_states, number_of_actions])
        self._P = P
        self.r = r
        self.prev_r = copy.copy(r)
        self._number_of_states = number_of_states
        self._number_of_actions = number_of_actions
        self._gamma = gamma
        self._max_iters = max_iters
        self.compute_values() # set Q-values 
        self._state = initial_state
        self._initial_state = initial_state 
        self._decayed = False 
        self._action = greedy(self._q[initial_state, :])

    @property
    def state_values(self):
        return np.max(self._q, axis=1)

    def compute_values(self, thresh=0.01):

        r = copy.copy(self.r)
        if len(r.shape) < 2:
            r = np.tile(r[:, None], [self._number_of_actions])

        assert r.shape == (self._number_of_states, self._number_of_actions), "invalid reward function"

        delta = np.inf
        k = 0
        while delta > thresh and k < self._max_iters:
            prev_q = np.copy(self._q)
            # apply bellman optimality operator
            self._q = r + self._gamma * self._P @ np.max(self._q, axis=-1) # S x A + (S x A x S) @ (S x 1)
            
            delta = np.max(np.abs(self._q - prev_q))

            k += 1


    def step(self, reward, discount, next_state):

        if (self.r != self.prev_r).any():
            self.compute_values() # recompute values if reward function changes

        # act greedily wrt true Q-values
        next_action = greedy(self._q[next_state, :])

        self.prev_r = copy.copy(self.r)


        return next_action


class FR(object):

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_state: int,
        sa: bool = False,
        policy: Callable = None,
        q: np.ndarray = None,
        step_size: float = 0.1) -> None:
        """tabular FR learning

        Args:
            number_of_states (int): size of state space
            number_of_actions (int): size of action space
            initial_state (int): index of initial state
            sa (bool, optional): whether to condition the FR on actions. defaults to False.
            policy (Callable, optional): a function which returns an action given Q-values. defaults to None.
            q (np.ndarray, optional): array of Q-values. defaults to None.
            step_size (float, optional): learning rate. defaults to 0.1.
        """
        if sa:
            self._F = np.zeros([number_of_states, number_of_actions, number_of_states])
            for a in range(number_of_actions): self._F[:, a, :] = np.eye(number_of_states);
        else:
            self._F = np.eye(number_of_states)
        self._sa = sa
        self._n = number_of_states
        self._number_of_actions = number_of_actions
        self.state_values = np.zeros([number_of_states])
        self._state = initial_state
        self._step_size = step_size
        self._initial_state = initial_state 
        self._episodes = -1
        self._policy = policy 
        self.td_errors = []
        self._q = q
        if self._policy is not None and self._q is not None: self._action = self._policy(self._q[initial_state, :]);
        else: self._action = 0; 

    @property
    def FR(self):
        return self._F


    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step and update the FR. 

        Args:
            reward (float)
            discount (float)
            next_state (int)

        Returns:
            int: An action in [0, 1, ..., |A|-1]
        """

        # if policy and q-function provided, select action with this
        if self._policy is not None and self._q is not None:
            next_action = self._policy(self._q[next_state, :])
        else:
            # return random action
            next_action = np.random.randint(self._number_of_actions)

        # update FR
        if self._sa: # if conditioning on actions
            delta = discount * self._F[next_state, next_action, :] - self._F[self._state, self._action, :]
            delta[self._state] = 0 # preserve diagonal
            self._F[self._state, self._action, :] += self._step_size * delta
        else:
            delta = discount * self._F[next_state, :] - self._F[self._state, :] # off-diagonal update
            delta[self._state] = 0 # preserve diagonal
            self._F[self._state, :] += self._step_size * delta

        # reset current state, action
        self._state = next_state
        self._action = next_action

        return next_action


class SR(object):

    def __init__(
        self,
        number_of_states: int,
        number_of_actions: int,
        initial_state: int,
        sa: bool = False,
        policy: Callable = None,
        q: np.ndarray = None,
        step_size: float = 0.1) -> None:
        """tabular SR learning

        Args:
            number_of_states (int): size of state space
            number_of_actions (int): size of action space
            initial_state (int): index of initial state
            sa (bool, optional): whether to condition on actions. defaults to False.
            policy (Callable, optional): function defining a policy over Q-values. defaults to None.
            q (np.ndarray, optional): Q-values. defaults to None.
            step_size (float, optional): learning rate. defaults to 0.1.
        """
        if sa: self._M = np.zeros([number_of_states, number_of_actions, number_of_states]);
        else: self._M = np.zeros([number_of_states, number_of_states]);
        self._sa = sa
        self._n = number_of_states
        self._number_of_actions = number_of_actions
        self.state_values = np.zeros([number_of_states])
        self._state = initial_state
        self._step_size = step_size
        self._initial_state = initial_state 
        self._policy = policy 
        self._q = q
        if self._policy is not None and self._q is not None: self._action = self._policy(self._q[initial_state, :]);
        else: self._action = 0; 

    @property
    def SR(self):
        return self._M


    def step(self, reward: float, discount: float, next_state: int) -> int:
        """Take a step and update the SR. 

        Args:
            reward (float)
            discount (float)
            next_state (int)

        Returns:
            int: Action in [0, 1, ..., |A|-1]
        """
            
        # if policy and q-function provided, select action with this
        if self._policy is not None and self._q is not None:
            #pdb.set_trace()
            next_action = self._policy(self._q[next_state, :])
        else:
            # return random action
            next_action = np.random.randint(self._number_of_actions)

        # compute SR update
        one_hot = np.eye(self._n)
        if self._sa:
            delta = one_hot[self._state] + discount * self._M[next_state, next_action, :] - self._M[self._state, self._action, :]
            self._M[self._state, self._action, :] += self._step_size * delta
        else:
            delta = one_hot[self._state] + discount * self._M[next_state, :] - self._M[self._state, :]
            self._M[self._state, :] += self._step_size * delta

        # reset current state, action
        self._state = next_state
        self._action = next_action

        return next_action

