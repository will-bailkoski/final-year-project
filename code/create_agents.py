from agent import Agent, IndependentQLearning, MARL_Comm
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


def convert_adj_to_power_graph(adj_table, gamma_hop, connection_slow=False):
    """Converts an adjacency table for a graph into an adjacency table for the corresponding power graph of distance
    gamma_hop

    Parameters
    -------------

    adj_table - The adjacency graph to convert

    gamma_hop - the amount the graph is allowed to change by

    connection_slow - False means the amount of jumps is factored in (so if the node is 2 jumps away its set to 2)

    Returns
    --------
    adjacency table of the power graph"""

    # If Gamma hop = 0 there is no communication. So [[0,0], [0,0]] for 2 agents
    if gamma_hop == 0:
        for i in range(len(adj_table)):
            for j in range(len(adj_table[i])):
                adj_table[i][j] = 0
        return adj_table

    # Goes through adj_table and adds to own dictionary the neighbours for each node
    neighbours = {z: [] for z in range(len(adj_table))}
    for i, row in enumerate(adj_table):
        for j, col in enumerate(row):
            if col != 0:
                i_neighbours = neighbours[i]
                i_neighbours.append((j, col))

    # Updates the adjacency table to be power graph.
    for k in range(gamma_hop - 1):
        new_table = deepcopy(adj_table)

        # Goes over each agent
        for i, row in enumerate(adj_table):
            # Goes for every distance for node i to node j
            for j, col in enumerate(row):
                if col != 0:
                    neighbours_of_j = neighbours[j]
                    for neighbour, dis in neighbours_of_j:
                        if adj_table[i][neighbour] == 0 and i != neighbour:
                            if connection_slow:
                                speed_of_others = adj_table[i][j]
                                new_table[i][neighbour] = dis + speed_of_others
                            else:
                                new_table[i][neighbour] = 1
        adj_table = new_table

    return adj_table

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
