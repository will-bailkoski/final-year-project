from adjacency import convert_adj_to_power_graph
from agent import Agent, IndependentQLearning
from ucb_marl_agent import MARL_Comm

from hyperparameters import graph_hyperparameters, AgentType


def create_agents(num_of_agents, agent_type, num_of_episodes=0, length_of_episode=0):
    """Creates the agents

    num_of_agents - The Number of agents to be created

    agent_type - The Type of the agent to be created

    num_of_episodes - The number of episodes to play

    length_of_episode - The length of the episode

    Returns
    A dictionary of agents of correct type
    """

    if agent_type == AgentType.IQL:
        return {f'agent_{i}': IndependentQLearning(f'agent_{i}', length_of_episode) for i in range(num_of_agents)}
    elif agent_type == AgentType.ORIGINAL:
        adj_table = graph_hyperparameters['graph']
        return create_marl_agents(num_of_agents, num_of_episodes, length_of_episode,
                                  graph_hyperparameters['gamma_hop'], adj_table,
                                  graph_hyperparameters['connection_slow'])
    else:
        return {f'agent_{i}': Agent(f'agent_{i}') for i in range(num_of_agents)}


def create_marl_agents(num_of_agents, num_of_episodes, length_of_episode, gamma_hop, adjacency_table, connection_slow):
    """
    Creates the MARL agents

    num_of_agents - The Number of agents to be created

    num_of_episodes - The number of episodes to play

    length_of_episode - The length of the episode

    gamma_hop - The gamma hop distance

    adjacency_table - The graph to be used

    connection_slow - Whether we want the connections to be instantaneous or whether a time delay should be incurred

    """

    agents = {f'agent_{i}': MARL_Comm(f'agent_{i}', num_of_agents, num_of_episodes, length_of_episode, gamma_hop) for i
              in range(num_of_agents)}

    power_graph = convert_adj_to_power_graph(adjacency_table, gamma_hop, connection_slow)
    print(power_graph)
    for i, row in enumerate(power_graph):
        for j, col in enumerate(row):
            if col != 0:
                agent_obj = agents[f'agent_{i}']
                agent_obj.update_neighbour(f'agent_{j}', col)

    return agents
