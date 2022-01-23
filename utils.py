import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import sys 
from scipy import stats

# custom libraries 
from envs import *


def epsilon_greedy(q_values, epsilon=0.1):
  """return an action epsilon-greedily from a vector of Q-values Q[s,:]"""
  if epsilon < np.random.random():
    return np.argmax(q_values)
  else:
    return np.random.randint(np.array(q_values).shape[-1])

def greedy(q_values, eps=None):
    """return the greedy action from a vector of Q-values"""
    return np.argmax(q_values)

def L1(v):
    return np.sum(np.abs(v))
    
def L2(v):
    return np.linalg.norm(v)

def sq_error(x, y):
    return (x - y)**2

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def flush_print(s):
    """
    print updates in a loop that refresh rather than stack 
    """
    print("\r" + s, end="")
    sys.stdout.flush()

def save_results(result_dict, name):
    """
    save experiment results to a pickle file 
    """
    with open('../saved/{}_results.pickle'.format(name), 'wb') as file:
        pickle.dump(result_dict, file)

def load_results(file_name):
    """
    load experiment results from a pickle file 
    """
    with open('{}.pickle'.format(file_name), 'rb') as file:
        return(pickle.load(file))


map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]

def plot_values(grid, values, ax=None, colormap='pink', vmin=0, vmax=10, title=None, cbar=True):
  if ax is None:
    #ax = plt.axes() 
      plt.imshow(values - 1000*(grid<0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
      plt.yticks([])
      plt.xticks([])
      if title is not None: plt.title(title, fontsize=14)
      if cbar: plt.colorbar(ticks=[vmin, vmax], orientation="horizontal"); 
  else:
      im = ax.imshow(values - 1000*(grid<0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
      ax.set_yticks([])
      ax.set_xticks([])
      #ax.grid()
      if title is not None: ax.set_title(title, fontsize=14)
      if cbar: plt.colorbar(im, ticks=[vmin, vmax], ax=ax, orientation="horizontal"); 

def plot_action_values(grid, action_values, vmin=-5, vmax=5):
  q = action_values
  fig = plt.figure(figsize=(10, 10))
  fig.subplots_adjust(wspace=0.3, hspace=0.3)
  for a in [0, 1, 2, 3]:
    ax = plt.subplot(4, 3, map_from_action_to_subplot(a))
    plot_values(grid, q[..., a], ax=None, vmin=vmin, vmax=vmax)
    action_name = map_from_action_to_name(a)
    plt.title(r"$q(x, \mathrm{" + action_name + r"})$")
    
  plt.subplot(4, 3, 5)
  v = np.max(q, axis=-1)
  plot_values(grid, v, colormap='summer', vmin=vmin, vmax=vmax)
  plt.title("$v(x)$")
  
  # Plot arrows:
  plt.subplot(4, 3, 11)
  plot_values(grid, grid>=0, vmax=1)
  for row in range(len(grid)):
    for col in range(len(grid[0])):
      if grid[row][col] >= 0:
        argmax_a = np.argmax(q[row, col])
        if argmax_a == 0:
          x = col
          y = row + 0.5
          dx = 0
          dy = -0.8
        if argmax_a == 1:
          x = col - 0.5
          y = row
          dx = 0.8
          dy = 0
        if argmax_a == 2:
          x = col
          y = row - 0.5
          dx = 0
          dy = 0.8
        if argmax_a == 3:
          x = col + 0.5
          y = row
          dx = -0.8
          dy = 0
        plt.arrow(x, y, dx, dy, width=0.02, head_width=0.4, head_length=0.4, length_includes_head=True, fc='k', ec='k')


def plot_actions(grid, action_values, ax=None, title=None, vmin=-5, vmax=5, c=['r']):
  if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(4, 4))
  q = action_values
  # Plot arrows:
  plot_values(grid, grid>=0, ax=ax, vmax=1, cbar=False)
  if len(c) != grid.size:
    #pdb.set_trace()
    colors = [c[0]] * grid.size
  else: colors = c; 
  colors = np.reshape(np.array(colors), grid.shape)
  idx = 0
  for row in range(len(grid)):
    for col in range(len(grid[0])):
      idx += 1
      if grid[row][col] >= 0:
        argmax_a = np.argmax(q[row, col])
        if argmax_a == 0:
          x = col
          y = row + 0.5
          dx = 0
          dy = -0.8
        if argmax_a == 1:
          x = col - 0.5
          y = row
          dx = 0.8
          dy = 0
        if argmax_a == 2:
          x = col
          y = row - 0.5
          dx = 0
          dy = 0.8
        if argmax_a == 3:
          x = col + 0.5
          y = row
          dx = -0.8
          dy = 0

        if argmax_a == 4: # up-right
          x = col - 0.5
          y = row + 0.5
          dx = 0.8 # go right
          dy = -0.8 # go up
        if argmax_a == 5: #down-right
          x = col - 0.5
          y = row - 0.5
          dx = 0.8 # go right
          dy = 0.8 # go down
        if argmax_a == 6: # down-left
          x = col + 0.5
          y = row - 0.5
          dx = -0.8 # go left
          dy = 0.8 # go down
        if argmax_a == 7: # up-left
          x = col + 0.5
          y = row + 0.5 
          dx = -0.8 # go left
          dy = -0.8 # go up
        
        ax.arrow(
          x, y, dx, dy,width=0.02, head_width=0.3, head_length=0.4,
          length_includes_head=True, fc=colors[row,col], ec=colors[row,col]
          )
  if title is not None: 
    ax.set_title(title, fontsize=30, c=colors[0, 0])

  h, w = grid.shape
  for y in range(h-1):
    ax.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
  for x in range(w-1):
    ax.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)



def plot_rewards(xs, rewards, color):
  mean = np.mean(rewards, axis=0)
  p90 = np.percentile(rewards, 90, axis=0)
  p10 = np.percentile(rewards, 10, axis=0)
  plt.plot(xs, mean, color=color, alpha=0.6)
  plt.fill_between(xs, p90, p10, color=color, alpha=0.3)

def parameter_study(parameter_values, parameter_name,
  agent_constructor, env_constructor, color, repetitions=10, number_of_steps=int(1e4)):
  mean_rewards = np.zeros((repetitions, len(parameter_values)))
  greedy_rewards = np.zeros((repetitions, len(parameter_values)))
  for rep in range(repetitions):
    for i, p in enumerate(parameter_values):
      env = env_constructor()
      agent = agent_constructor()
      if 'eps' in parameter_name:
        agent.set_epsilon(p)
      elif 'alpha' in parameter_name:
        agent._step_size = p
      else:
        raise NameError("Unknown parameter_name: {}".format(parameter_name))
      mean_rewards[rep, i] = run_experiment(env, agent, number_of_steps)
      agent.set_epsilon(0.)
      agent._step_size = 0.
      greedy_rewards[rep, i] = run_experiment(env, agent, number_of_steps//10)
      del env
      del agent

  plt.subplot(1, 2, 1)
  plot_rewards(parameter_values, mean_rewards, color)
  plt.yticks=([0, 1], [0, 1])
  plt.ylabel("Average reward over first {} steps".format(number_of_steps), size=12)
  plt.xlabel(parameter_name, size=12)

  plt.subplot(1, 2, 2)
  plot_rewards(parameter_values, greedy_rewards, color)
  plt.yticks=([0, 1], [0, 1])
  plt.ylabel("Final rewards, with greedy policy".format(number_of_steps), size=12)
  plt.xlabel(parameter_name, size=12)


def colorline(x, y, z):
    """
    Based on:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=plt.get_cmap('copper_r'),
                              norm=plt.Normalize(0.0, 1.0), linewidth=3)

    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plotting_helper_function(_x, _y, title=None, ylabel=None):
  z = np.linspace(0, 0.9, len(_x))**0.7
  colorline(_x, _y, z)
  plt.plot(0, 0, '*', color='#000000', ms=20, alpha=0.7, label='$w^*$')
  plt.plot(1, 1, '.', color='#ee0000', alpha=0.7, ms=20, label='$w_0$')
  min_y, max_y = np.min(_y), np.max(_y)
  min_x, max_x = np.min(_x), np.max(_x)
  min_y, max_y = np.min([0, min_y]), np.max([0, max_y])
  min_x, max_x = np.min([0, min_x]), np.max([0, max_x])
  range_y = max_y - min_y
  range_x = max_x - min_x
  max_range = np.max([range_y, range_x])
  plt.arrow(_x[-3], _y[-3], _x[-1] - _x[-3], _y[-1] - _y[-3], color='k',
            head_width=0.04*max_range, head_length=0.04*max_range,
            head_starts_at_zero=False)
  plt.ylim(min_y - 0.2*range_y, max_y + 0.2*range_y)
  plt.xlim(min_x - 0.2*range_x, max_x + 0.2*range_x)
  ax = plt.gca()
  ax.ticklabel_format(style='plain', useMathText=True)
  plt.legend(loc=2)
  plt.xticks(rotation=12, fontsize=10)
  plt.yticks(rotation=12, fontsize=10)
  plt.locator_params(nbins=3)
  if title is not None:
    plt.title(title, fontsize=20)
  if ylabel is not None:
    plt.ylabel(ylabel, fontsize=20)
  

def errorfill(x, y, yerr, color='C0', alpha_fill=0.3, ax=None, label=None, lw=1, marker=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, lw=lw, marker=marker)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.7)
    #ax.legend(fontsize=13)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def multicolor_label(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_of_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.17, -0.4),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.10, 0.2), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

def runtest(alg1, alg2):
    t1 = stats.ttest_ind(alg1, alg2, equal_var=False)
    #print(f"Env = {envname}")
    print(f"Alg1 vs. Alg2, tstat = {np.round(t1[0], 2)}, p-value = {np.round(t1[1], 6)}")