from copy import deepcopy


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
    for k in range(gamma_hop-1):                 
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
                                new_table[i][neighbour] = dis+speed_of_others
                            else:
                                new_table[i][neighbour] = 1
        adj_table = new_table

    return adj_table
