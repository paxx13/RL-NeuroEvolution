# RL-NeuroEvolution

the program trains an RL agent for the 'MountainCarContinuous-v0' gym environment using NeuroEvolution of Augmenting Topologies(NEAT). The implementation is based on the [neat-python](https://github.com/CodeReclaimers/neat-python) framework 

usage:
```sh
python .\main.py -h
usage: main.py [-h] [-t] [-c CHECKPOINT]

evolves a policy for solving gym 'MountainCarContinuous-v0' environment using
neural evolution augmented topologies (NEAT)

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           set to learn a policy
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint which is used for training further or to
                        run
```