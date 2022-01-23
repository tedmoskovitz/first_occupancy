## The First-Occupancy Representation

Implementation for our paper [A First-Occupancy Representation for Reinforcement Learning](https://openreview.net/forum?id=JBAZe2yN6Ub).



dependencies: `numpy`, `scipy`, `matplotlib`, `seaborn`

files: 
- `agents.py`: basic agent classes for GPI, value iteration, the FR, and the SR
- `envs.py`: FourRoom and Escape environment classes
- `frp.py`: FR planning (FRP) agent class
- `utils.py`: basic helper functions
- `runners.py`: functions for running experiments
- `four_rooms.ipynb`: result notebook



If you find this code useful, please cite using:

```
@misc{moskovitz2021firstoccupancy,
      title={A First-Occupancy Representation for Reinforcement Learning}, 
      author={Ted Moskovitz and Spencer R. Wilson and Maneesh Sahani},
      year={2021},
      eprint={2109.13863},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

