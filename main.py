import numpy as np
import gym
import neat
import argparse

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
    
CONFIG = "./config"
MAX_STEPS = 1000   # maximum episode steps
EVAL_RUNS = 100    # amount of evaluation runs


def run_episode(env, policy, render=False):
    state = env.reset()
    total_reward = 0
    for s in range(MAX_STEPS):
        if render:
            env.render()
        action = policy.activate(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        policy = neat.nn.FeedForwardNetwork.create(genome, config)
        r = []
        for ep in range(EVAL_RUNS): 
            r.append(run_episode(env, policy))
        genome.fitness = np.mean(r)


def train(population):
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.Checkpointer(5))

    population.run(eval_genomes, 200)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='evolves a policy for solving gym \'MountainCarContinuous-v0\' environment using NeuroEvolution of Augmenting Topologies(NEAT)')
    arg_parser.add_argument('-t', '--train', help='set to learn a policy', action="store_true")
    arg_parser.add_argument('-c', '--checkpoint', help='checkpoint which is used for training further or to run', dest="checkpoint")
    args = arg_parser.parse_args()
    
    
    if args.train: 
        if args.checkpoint == None:
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
            p = neat.Population(config)
        else:
            p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + args.checkpoint)
        train(p)
    else:
        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + args.checkpoint)
        winner = p.run(eval_genomes, 1)
        policy = neat.nn.FeedForwardNetwork.create(winner, p.config)        
        run_episode(env, policy, render=True)
