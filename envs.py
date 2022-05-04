import matplotlib.pyplot as plt
import numpy as np
from utils import plot_actions


class Escape(object):

  def __init__(self, size=21, start_state=None, barrier=True, noisy=False, seed=7697, discount=0.99):
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward, episode will terminate
    W = -1
    G = 20
    np.random.seed(seed)
    self._W = W  # wall
    self._G = G  # goal
    self.discount = discount

    assert size >= 9, "too small"
    self._layout = np.zeros([size, size])
    self._layout[0, :] = W # top wall
    self._layout[-1, :] = W # bottom wall
    self._layout[:, 0] = W # left
    self._layout[:, -1] = W # right
    #self._layout[-2, size // 2 - 1: size // 2 + 2] = G
    self._layout[-2, size// 2] = G
    self.r = np.reshape(self._layout > 0, [-1])
    self.r = self.r.astype(np.float32)
    if barrier: self._layout[size // 2, size // 4 : 3 * size // 4 + 1] = W; 


    if start_state is None: start_state = self.get_obs(s=(1, size // 2)); 
    self._start_state = self.obs_to_state_coords(start_state)
    self._start_obs = start_state
    self._episodes = 0
    self._state = self._start_state
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    # reward
    flat_layout = self._layout.flatten()
    wall_idxs = np.stack(np.where(flat_layout == W)).T

    lat_layout = self._layout.flatten()
    wall_idxs = np.stack(np.where(flat_layout == W)).T
    # possible reward states are those where there isn't a wall
    self._free_states = np.array([s for s in range(len(flat_layout)) if s not in wall_idxs])

    self._noisy = noisy

    

  @property
  def number_of_states(self):
      return self._number_of_states


  def get_obs(self, s=None):
    y, x = self._state if s is None else s
    return y*self._layout.shape[1] + x

  def obs_to_state(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    s = np.copy(self._layout)
    s[y, x] = 4
    return s

  def obs_to_state_coords(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    return (y, x)

  @property
  def episodes(self):
      return self._episodes


  def reset(self):
    self._state = self._start_state
    self._episodes = 0
    y, x = self._state


    return self._layout[y, x], self.discount, self.get_obs(), False

  def step(self, action):
    done = False
    y, x = self._state
    
    if action == 0:  # up
      new_state = (y - 1, x)
    elif action == 1:  # right
      new_state = (y, x + 1)
    elif action == 2:  # down
      new_state = (y + 1, x)
    elif action == 3:  # left
      new_state = (y, x - 1)
    elif action == 4: # up and right
      new_state = (y - 1, x + 1)
    elif action == 5: # down and right
      new_state = (y + 1, x + 1)
    elif action == 6: # down and left
      new_state = (y + 1, x - 1)
    elif action == 7: # up and left
      new_state = (y - 1, x - 1)
    else:
      raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

    new_y, new_x = new_state
    reward = self._layout[new_y, new_x]
    if self._layout[new_y, new_x] == self._W:  # wall
      discount = self.discount
      new_state = (y, x)
    elif self._layout[new_y, new_x] == 0:  # empty cell
      discount = self.discount
    else:  # a goal
      discount = self.discount
      self._episodes += 1
      done = True

    if self._noisy:
      width = self._layout.shape[1]
      reward += 10*np.random.normal(0, width - new_x + new_y)

    self._state = new_state

    return reward, discount, self.get_obs(), done

  def plot_grid(
    self,
    traj=None, M=None, ax=None, goals=None,
    cbar=False, traj_color="C2", title='Escape', show_idxs=False, show_grid=True, flip=False
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
    
    ax.imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    h, w = self._layout.shape
    startx, starty = self._start_state
    for gcol in range(h // 2 - 1, h // 2 + 2):
      ax.text(gcol, h-2, r"$\mathbf{g}$", ha='center', va='center', fontsize=13, color='C2')

    if show_idxs:
      for i in range(self._layout.shape[0]):
          for j in range(self._layout.shape[1]):
              ax.text(j, i, f"{self.get_obs(np.array([i, j]))}", ha='center', va='center', fontsize=6)
    
    if show_grid:
      for y in range(h-1):
        ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if traj is not None:
      # plot trajectory, list of [(y0, x0), (y1, x1), ...]
      if goals is None:
        traj = np.vstack(traj)
        ax.plot(traj[:, 1] if not flip else 24-traj[:, 1], traj[:, 0], c=traj_color, lw=3)
      else:
        # draw goals
        for i,g in enumerate(goals):
          if g != np.argmax(self.r):
            y, x = self.obs_to_state_coords(g)
            ax.text(x, y, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=16, color=f'C{i}')
        # draw trajectories
        traj = np.vstack(traj)
        ax.plot(traj[:, 1], traj[:, 0], c=traj_color, lw=3, ls='-')

    if M is not None:
      # M is either a vector of len |S| of a matrix of size |A| x |S|
      if len(M.shape) == 1:
        M_2d = M.reshape(h, w)
      else:
        M_2d = np.mean(M, axis=0).reshape(h, w)

      ax.imshow(M_2d, cmap='viridis', interpolation='nearest')
      if cbar: ax.colorbar(); 

  def plot_planning_output(self, pi, s_ast, ax=None, colors=None, show_states=False, suptitle=None, Pi=None):
    if ax is None:
        n = 2 if not show_states else 3
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
        fig.suptitle(suptitle, fontsize=23)
        
    
    pi2c = None
    if colors is not None:
      assert len(colors) == len(np.unique(pi)), "incompatible number of colors";
      pi2c = dict(zip(np.unique(pi), colors))

    axs[0].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[1].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    #ax.grid(1)
    axs[0].set_xticks([]); axs[1].set_xticks([])
    axs[0].set_yticks([]); axs[1].set_yticks([])
    axs[0].set_title(r"$\pi_F(s)$", fontsize=16)
    axs[1].set_title(r"$s_F(s)$", fontsize=16)
    h, w = self._layout.shape
    color_list = []
    for y in range(h):
      for x in range(w):
        pidx = pi[y, x]
        if self._layout[y, x] >= 0:
          c = 'k' if pi2c is None else pi2c[pidx]
          if Pi is None:
            axs[0].text(x, y, r"$\pi_{}$".format(pidx), ha='center', va='center', color=c, fontsize=16)
          
          
          row, col = self.obs_to_state_coords(s_ast[y, x])
          axs[1].text(x, y, "{},{}".format(row-1, col-1), ha='center', va='center', color=c, fontsize=10) #.format(s_ast[y, x])

    # plot arrows 
    if Pi is not None:
      # construct "composite" q-function
      cq = np.zeros([self._number_of_states, 4])
      color_list = []
      for s in range(self._number_of_states):
        pidx = np.reshape(pi, [-1])[s]
        cq[s] = Pi[pidx].q[s, :]
        color_list.append(pi2c[pidx])
      
      plot_actions(self._layout, cq.reshape(self._layout.shape + (4,)), ax=axs[0], c=color_list)


    h, w = self._layout.shape
    for y in range(h-1):
      axs[0].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      axs[1].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      axs[0].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      axs[1].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if show_states: 
      axs[2].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
      #ax.grid(1)
      axs[2].set_xticks([]); axs[1].set_xticks([])
      axs[2].set_yticks([]); axs[1].set_yticks([])
      axs[2].set_title(r"$\mathcal{S}$", fontsize=20)
      for i in range(1, self._layout.shape[0]-1):
          for j in range(1, self._layout.shape[1]-1):
              axs[2].text(j, i, "{},{}".format(i-1, j-1), ha='center', va='center', color='k', fontsize=10) 

      h, w = self._layout.shape
      for y in range(h-1):
        axs[2].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        axs[2].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)


class FourRooms(object):

  def __init__(self, start_state=100, reset_goal=False, noise=None, seed=7697, discount=0.95):
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward
    W = -1
    G = 1
    np.random.seed(seed)
    self._W = W  # wall
    self._G = G  # goal
    self.discount = discount

    self._layout = np.array([
        [W, W, W, W, W, W, W, W, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W],
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, W, W, W, W, W, 0, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, W, W, W, W, W, W, W, W, W, W], 
    ])

    self._reset_goal = reset_goal
    self._start_state = self.obs_to_state_coords(start_state)
    self._episodes = 0
    self._state = self._start_state
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    # reward
    flat_layout = self._layout.flatten()
    wall_idxs = np.stack(np.where(flat_layout == W)).T
    # possible reward states are those where there isn't a wall
    self._possible_reward_states = np.array([s for s in range(len(flat_layout)) if s not in wall_idxs])
    self.r = np.zeros(self._number_of_states)
    goal_state = np.random.choice(self._possible_reward_states)
    self._goal_hist = [goal_state] 
    self.r[goal_state] = 50
    self._max_goals = 5
    self._goals_reached = 0
    self._noise = noise
    self._switch_steps = []
    self._steps = 0


    # transition matrix
    self._R = np.array([-1, 0, 50])
    P = np.zeros([self._number_of_states, 4, self._number_of_states])
    l = self._layout.shape[0]

    if self._noise is not None:
      eps = self._noise
      p = 1 - eps
    else:
      p = 1
    for a in range(4): 
      for s in range(self._number_of_states):
        for sp in range(self._number_of_states):
          
          if a == 0: 
            if sp == s - l and flat_layout[sp] != W: P[s, a, sp] =  p; 
            elif sp == s - l and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 1: 
            if sp == s + 1 and flat_layout[sp] != W: P[s, a, sp] = p; 
            elif sp == s + 1 and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 2: 
            if sp == s + l and flat_layout[sp ] != W: P[s, a, sp] = p; 
            elif sp == s + l and flat_layout[sp] == W: P[s, a, s] = p;
          else: 
            if sp == s - 1 and flat_layout[sp] != W: P[s, a, sp] = p;
            elif sp == s - 1 and flat_layout[sp] == W: P[s, a, s] = p;


    if self._noise is not None:
      eps = self._noise
      p = 1 - eps
      f = 4
      for a in range(4): 
        for s in range(self._number_of_states):
          if s - l > 0: # up
            if flat_layout[s - l] == W: P[s, a, s] += eps / f;
            else: P[s, a, s - l] += eps / f; 
          if s + l < len(flat_layout):
            if flat_layout[s + l] == W: P[s, a, s] += eps / f;
            else: P[s, a, s + l] += eps / f; 
          if s - 1 > 0: 
            if flat_layout[s - 1] == W: P[s, a, s] += eps / f;
            else: P[s, a, s - 1] += eps / f; 
          if s + 1 < len(flat_layout): 
            if flat_layout[s + 1] == W: P[s, a, s] += eps / f;
            else: P[s, a, s + 1] += eps / f; 
      
    self._P = P
    

  @property
  def number_of_states(self):
      return self._number_of_states

  @property
  def goal_states(self):
      return self._goal_hist

  def get_obs(self, s=None):
    y, x = self._state if s is None else s
    return y*self._layout.shape[1] + x

  def obs_to_state(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    s = np.copy(grid._layout)
    s[y, x] = 4
    return s

  def obs_to_state_coords(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    return (y, x)

  @property
  def episodes(self):
      return self._episodes

  def P(self, s, a, sp, r):
    if r not in self._R: return 0; 
    r_idx = np.where(self._R == r)[0][0]
    return self._P[s, a, sp, r_idx]


  def reset(self):
    self._state = self._start_state
    self._episodes = 0
    y, x = self._state
    self._switch_steps = [] # track when goals switch
    self.r = np.zeros(self._number_of_states)
    goal_state = np.random.choice(self._possible_reward_states)
    self._goal_hist = [goal_state] 
    self.r[goal_state] = 50
    self._max_goals = 5
    self._goals_reached = 0
    self._switch_steps = []
    self._steps = 0
    return self._layout[y, x], 0.9, self.get_obs(), False

  def step(self, action):
    done = False
    y, x = self._state
    r2d = np.reshape(self.r, self._layout.shape)

    if self._noise is not None:
      assert self._noise <= 1 and self._noise >= 0, "invalid noise setting"
      if np.random.random() < self._noise:
        action = np.random.choice(4)
      else:
        action = action
    
    if action == 0:  # up
      new_state = (y - 1, x)
    elif action == 1:  # right
      new_state = (y, x + 1)
    elif action == 2:  # down
      new_state = (y + 1, x)
    elif action == 3:  # left
      new_state = (y, x - 1)
    else:
      raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

    new_y, new_x = new_state
    reward = self._layout[new_y, new_x]
    if self._layout[new_y, new_x] == self._W:  # wall
      discount = self.discount
      new_state = (y, x)
    elif self._layout[new_y, new_x] == 0 and r2d[new_y, new_x] == 0:  # empty cell
      discount = self.discount
    else:  # a goal
      discount = self.discount
      self._episodes += 1
      reward = r2d[new_y, new_x]
      if self._reset_goal:
        
        self.r = np.zeros(self._number_of_states)
        self.r[np.random.choice(self._possible_reward_states)] = 50
        self._goal_hist.append(np.argmax(self.r))
        self._switch_steps.append(self._steps)

      self._goals_reached += 1

      if self._goals_reached >= self._max_goals:
        done = True



    self._state = new_state
    self._steps += 1
    return reward, discount, self.get_obs(), done

  def plot_grid(
    self,
    traj=None, M=None, ax=None, goals=None,
    cbar=False, traj_color="C2", title='FourRooms', show_idxs=False
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
    
    ax.imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    startx, starty = self._start_state
    goalx, goaly = self.obs_to_state_coords(np.argmax(self.r))
    ax.text(starty, startx, r"$\mathbf{s_0}$", ha='center', va='center', fontsize=16)
    if traj is None:
      ax.text(goaly, goalx, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=16)
    
    if show_idxs:
      for i in range(self._layout.shape[0]):
          for j in range(self._layout.shape[1]):
              ax.text(j, i, f"{self.get_obs(np.array([i, j]))}", ha='center', va='center')
    
    h, w = self._layout.shape
    for y in range(h-1):
      ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if traj is not None:
      # plot trajectory, list of [(y0, x0), (y1, x1), ...]
      if goals is None:
        traj = np.vstack(traj)
        ax.plot(traj[:, 1], traj[:, 0], c=traj_color, lw=3)
      else:
        # draw goals
        for i,g in enumerate(goals):
          if g != np.argmax(self.r):
            y, x = self.obs_to_state_coords(g)
            ax.text(x, y, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=16, color=f'C{i}')
        # draw trajectories
        traj = np.vstack(traj)
        ax.plot(traj[:, 1], traj[:, 0], c=traj_color, lw=3, ls='-')

    if M is not None:
      # M is either a vector of len |S| of a matrix of size |A| x |S|
      if len(M.shape) == 1:
        M_2d = M.reshape(h, w)
      else:
        M_2d = np.mean(M, axis=0).reshape(h, w)

      ax.imshow(M_2d, cmap='viridis', interpolation='nearest')
      if cbar: ax.colorbar(); 

  def plot_planning_output(self, pi, s_ast, ax=None, colors=None, show_states=False, suptitle=None, Pi=None):
    if ax is None:
        n = 2 if not show_states else 3
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
        fig.suptitle(suptitle, fontsize=23)
    
    pi2c = None
    if colors is not None:
      assert len(colors) == len(np.unique(pi)), "incompatible number of colors";
      pi2c = dict(zip(np.unique(pi), colors))

    axs[0].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[1].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[0].set_xticks([]); axs[1].set_xticks([])
    axs[0].set_yticks([]); axs[1].set_yticks([])
    axs[0].set_title(r"$\pi_F(s)$", fontsize=16)
    axs[1].set_title(r"$s_F(s)$", fontsize=16)
    h, w = self._layout.shape
    color_list = []
    for y in range(h):
      for x in range(w):
        pidx = pi[y, x]
        if self._layout[y, x] >= 0:
          c = 'k' if pi2c is None else pi2c[pidx]
          if Pi is None:
            axs[0].text(x, y, r"$\pi_{}$".format(pidx), ha='center', va='center', color=c, fontsize=16)
          row, col = self.obs_to_state_coords(s_ast[y, x])
          axs[1].text(x, y, "{},{}".format(row-1, col-1), ha='center', va='center', color=c, fontsize=10) #.format(s_ast[y, x])

    # plot arrows 
    if Pi is not None:
      # construct "composite" q-function
      cq = np.zeros([self._number_of_states, 4])
      color_list = []
      for s in range(self._number_of_states):
        pidx = np.reshape(pi, [-1])[s]
        cq[s] = Pi[pidx].q[s, :]
        color_list.append(pi2c[pidx])
      
      plot_actions(self._layout, cq.reshape(self._layout.shape + (4,)), ax=axs[0], c=color_list)


    h, w = self._layout.shape
    for y in range(h-1):
      axs[0].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      axs[1].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      axs[0].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      axs[1].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if show_states: 
      axs[2].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
      axs[2].set_xticks([]); axs[1].set_xticks([])
      axs[2].set_yticks([]); axs[1].set_yticks([])
      axs[2].set_title(r"$\mathcal{S}$", fontsize=20)
      for i in range(1, self._layout.shape[0]-1):
          for j in range(1, self._layout.shape[1]-1):
              axs[2].text(j, i, "{},{}".format(i-1, j-1), ha='center', va='center', color='k', fontsize=10) #.format(idx)

      h, w = self._layout.shape
      for y in range(h-1):
        axs[2].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        axs[2].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)


class BasicGrid(object):

  def __init__(self, start_state=25, noisy=False):
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward, episode will terminate
    W = -1
    G = 1
    self._W = W  # wall
    self._G = G  # goal
    self._layout = np.array([
        [W, W, W, W, W, W],
        [W, 0, 0, 0, 0, W],
        [W, 0, 0, G, 0, W], 
        [W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W], 
        [W, W, W, W, W, W]
    ])

    self._start_state = self.obs_to_state_coords(start_state)#(4, 1)
    self._episodes = 0
    self._state = self._start_state
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    self._noisy = noisy

  @property
  def number_of_states(self):
      return self._number_of_states

  def get_obs(self, s=None):
    y, x = self._state if s is None else s
    return y*self._layout.shape[1] + x

  def obs_to_state(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    s = np.copy(grid._layout)
    s[y, x] = 4
    return s

  def obs_to_state_coords(self, obs):
    x = obs % self._layout.shape[1]
    y = obs // self._layout.shape[1]
    return (y, x)

  @property
  def episodes(self):
      return self._episodes

  def reset(self):
    self._state = self._start_state
    self._episodes = 0
    y, x = self._state
    return self._layout[y, x], 0.9, self.get_obs(), False

  def step(self, action):
    done = False
    y, x = self._state
    
    if action == 0:  # up
      new_state = (y - 1, x)
    elif action == 1:  # right
      new_state = (y, x + 1)
    elif action == 2:  # down
      new_state = (y + 1, x)
    elif action == 3:  # left
      new_state = (y, x - 1)
    else:
      raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

    new_y, new_x = new_state
    reward = self._layout[new_y, new_x]
    if self._layout[new_y, new_x] == self._W:  # wall
      discount = 0.9
      new_state = (y, x)
    elif self._layout[new_y, new_x] == 0:  # empty cell
      discount = 0.9
    else:  # a goal
      discount = 0.9
      self._episodes += 1
      done = True

    if self._noisy:
      width = self._layout.shape[1]
      reward += 10*np.random.normal(0, width - new_x + new_y)

    self._state = new_state
    return reward, discount, self.get_obs(), done

  def plot_grid(self, traj=None, M=None, ax=None, cbar=False, traj_color=["C2"], title='A Grid'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    im = ax.imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    ax.text(1, 4, r"$\mathbf{s_0}$", ha='center', va='center', fontsize=22)
    ax.text(3, 2, r"$\mathbf{s_g}$", ha='center', va='center', fontsize=22)

    h, w = self._layout.shape
    for y in range(h-1):
      ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if traj is not None:
      # plot trajectory, list of [(y0, x0), (y1, x1), ...]
      traj = np.vstack(traj)
      ax.plot(traj[:, 1], traj[:, 0], c=traj_color[0], lw=3)

    if M is not None:
      # M is either a vector of len |S| of a matrix of size |A| x |S|
      if len(M.shape) == 1:
        M_2d = M.reshape(h, w)
      else:
        M_2d = np.mean(M, axis=0).reshape(h, w)

      im = ax.imshow(M_2d, cmap='viridis', interpolation='nearest')
      if cbar: ax.colorbar(); 

    return im

  def plot_planning_output(self, pi, s_ast, ax=None, colors=None, show_states=False, Pi=None):
    if ax is None:
        n = 2 if not show_states else 3
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    
    pi2c = None
    if colors is not None:
      assert len(colors) == len(np.unique(pi)), "incompatible number of colors";
      pi2c = dict(zip(np.unique(pi), colors))

    axs[0].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[1].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
    axs[0].set_xticks([]); axs[1].set_xticks([])
    axs[0].set_yticks([]); axs[1].set_yticks([])
    axs[0].set_title(r"$\pi^F(s)$", fontsize=30)
    axs[1].set_title(r"$s^F(s)$", fontsize=30)
    s_ast[1, :] = 15
    s_ast[2, :] = 15
    s_ast[4, 3] = 15
    pi[2, 4] = 0
    pi[1, 3] = 0
    
    for y in range(1, 5):
      for x in range(1, 5):
        pidx = pi[y, x]
        c = 'k' if pi2c is None else pi2c[pidx]
        if Pi is None:
          axs[0].text(y, x, r"$\pi_{}$".format(pidx), ha='center', va='center', color=c, fontsize=22)
        row, col = self.obs_to_state_coords(s_ast[y, x])
        
        axs[1].text(x, y, "{},{}".format(row-1, col-1), ha='center', va='center', color=c, fontsize=22) 


    # plot arrows 
    if Pi is not None:
      # construct "composite" q-function
      cq = np.zeros([self._number_of_states, 4])
      color_list = []
      for s in range(self._number_of_states):
        pidx = np.reshape(pi, [-1])[s]
        cq[s] = Pi[pidx].q[s, :]
        color_list.append(pi2c[pidx])
      
      plot_actions(self._layout, cq.reshape(self._layout.shape + (4,)), ax=axs[0], c=color_list)

    h, w = self._layout.shape
    for y in range(h-1):
      axs[0].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      axs[1].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      axs[0].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      axs[1].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    if show_states: 
      axs[2].imshow(self._layout >= 0, interpolation="nearest", cmap='pink')
      axs[2].set_xticks([]); axs[1].set_xticks([])
      axs[2].set_yticks([]); axs[1].set_yticks([])
      axs[2].set_title(r"$\mathcal{S}$", fontsize=26)
      for y in range(1, 5):
        for x in range(1, 5):
          idx = 6 * y + x 
          axs[2].text(x, y, "{},{}".format(y-1, x-1), ha='center', va='center', color='k', fontsize=22) #.format(idx)

      h, w = self._layout.shape
      for y in range(h-1):
        axs[2].plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
      for x in range(w-1):
        axs[2].plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)


class RiverSwim(object):
    
    def __init__(self):
        
        self._num_actions = 2
        self._num_states = 6
        self._start_state = np.random.choice([1, 2], p=[0.5, 0.5])
        self._state = self._start_state
        
        # define transition matrices for each state, |A|(2) x |S|(6)
        # first row is action = 0 (left), second is action = 1 (right)
        Ps0 = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0, 0.0, 0.0]
        ]) # from state 0, if you go left, you stay in state 0, if you go right, you escape 
        # with 30% probability 
        Ps1 = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.6, 0.3, 0.0, 0.0, 0.0],
        ])
        Ps2 = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.2, 0.7, 0.0, 0.0]
        ])
        Ps3 = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.6, 0.3, 0.0]
        ])
        Ps4 = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.6, 0.3]
        ])
        Ps5 = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.6]
        ])
        self.P = np.stack([Ps0, Ps1, Ps2, Ps3, Ps4, Ps5]) # |S| x |A| x |S'|
        
        # reward function r(s, a, s')
        self.r = np.zeros([self._num_states, self._num_actions, self._num_states])
        self.r[0, 0, 0] = 5
        self.r[5, 1, 5] = 1e4
        self.discount = 0.95
        self.S = np.arange(self._num_states)
    
    def get_obs(self, s=None):
        return s if s is not None else self._state
    
    @property
    def size(self):
        return self._num_states
    
    def obs_to_state(self, obs):
        return obs
    
    def reset(self):
        self._state = np.random.choice([1, 2], p=[0.5, 0.5])
        
        return 0, self.discount, self.get_obs(), False
    
    def step(self, action):
        assert action in [0, 1], "invalid action"
        
        done = False
        # get (s, a) -> s' transition probs
        transition_probs = self.P[self._state, action, :] # |S'| vector
        # get next state
        next_state = np.random.choice(self.S, p=transition_probs)
        # get reward
        reward = self.r[self._state, action, next_state]
        # reset state
        self._state = next_state
        
        return reward, self.discount, self.get_obs(), done
        
