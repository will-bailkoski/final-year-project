from hyperparameters import reward_function


def final_reward(agent_old_reward):
    """
    This is the actual reward function to be used

    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns 
    The reward 
    """
    if reward_function['reward'] == 'mean':
        return _mean_reward(agent_old_reward)

    if reward_function['reward'] == 'split':
        return switch_reward(agent_old_reward)


def _mean_reward(agent_old_reward):
    """
    This will be the reward function for to be used for the average reward over all agents
    
    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns
    The mean reward
    """

    total_reward = 0
    for reward_per_agent in agent_old_reward.values():
        total_reward += reward_per_agent

    return total_reward / len(agent_old_reward.keys())


def switch_reward(agent_old_reward):
    """
    This will average out the reward for agents 0 and 3 (A & D).

    agents_old_reward - A dictionary of rewards with keys being agent names

    Returns
    The mean reward of agents 0 and 3

    """

    agent_one = agent_old_reward['agent_0']
    agent_four = agent_old_reward['agent_3']

    return (agent_four + agent_one) / 2
