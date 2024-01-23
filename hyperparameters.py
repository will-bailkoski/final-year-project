"""In this file, we list sets of hyperparameters. The selection for which hyperparameters to use for each part of the
process can be found at the bottom of the file."""
from enum import Enum


class AgentType(Enum):
    """Specifies the different agent types used"""

    RANDOM = 'RANDOM'
    IQL = 'IQL'
    ORIGINAL = 'ORIGINAL'


# GRAPH
# Adjacency graph & connection slow & gamma hop
multiple_graph_parameters = [
    {
        'graph': lambda x: [[0 if i == j else 1 for i in range(x)] for j in range(x)],
        'connection_slow': False,
        'gamma_hop': 1
    },
    {
        'graph': [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        'connection_slow': True,
        'gamma_hop': 0
    },
    {
        'graph': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        'connection_slow': True,
        'gamma_hop': 3
    },
    {
        'graph': [[0]],
        'connection_slow': True,
        'gamma_hop': 3
    }
]

# AGENT
# What agent to use
# Number of agents
# Size of state space
agent_multiple_parameters = [
    {
        'agent_choice': AgentType.RANDOM,
        'num_of_agents': 4,
        'size_of_state_space': 10 ** 4
    },
    {
        'agent_choice': AgentType.IQL,
        'num_of_agents': 4,
        'size_of_state_space': 10 ** 4
    },
    {
        'agent_choice': AgentType.ORIGINAL,
        'num_of_agents': 4,
        'size_of_state_space': 10 ** 4
    },
]

# IQL
# Gamma, alpha, Greedy
iql_multiple_parameters = [
    {
        'gamma': 0.99, #0.99, #0.8,
        'greedy': 0.9, #0.9,
        'alpha': 0.01, #0.01, #0.8
    }
]

# UCB
# c value
# prob value
ucb_marl_multiple_parameters = [
    {
        'c': 0.02,
        'probability': 0.1,
    }
]

# EVALUATION
# NUMBER_OF_TRIALS
# NUM_EVALUATION_EPISODES
# EVALUATION_INTERVALS
evaluation_multiple_parameters = [
    {
        'num_of_trials': 5,
        'num_evaluation_episodes': 16,
        'evaluation_interval': 1,
    }
]

# REWARD
# Reward function to use
reward_multiple_parameters = [
    {
        'reward': 'mean',
    },
    {
        'reward': 'split'
    }
]

# TRAIN
# NUM_OF_CYCLES = 100
# NUM_OF_EPISODES
# LOCAL_RATIO = 0
train_multiple_parameters = [
    {
        'num_of_episodes': 1000,
        'num_of_cycles': 10,
        'local_ratio': 0,

    }
]

switch_multiple_parameters = [
    {
        'switch': False
    }
]

# Hyperparameter Selection

graph_hyperparameters = multiple_graph_parameters[1]

agent_hyperparameters = agent_multiple_parameters[2]

iql_hyperparameters = iql_multiple_parameters[0]

ucb_marl_hyperparameters = ucb_marl_multiple_parameters[0]

evaluation_hyperparameters = evaluation_multiple_parameters[0]

reward_function = reward_multiple_parameters[0]

train_hyperparameters = train_multiple_parameters[0]

switch_hyperparameters = switch_multiple_parameters[0]
