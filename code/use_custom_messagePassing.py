"""This will create the necessary environment for use of the rest of the program.  Plug & play with this
abstracting which environment is used"""
from pettingZoo.PettingZoo.pettingzoo.mpe.custom_env.custom_env import env, parallel_env
from create_agents import create_marl_agents
from hyperparameters import train_hyperparameters, agent_hyperparameters, AgentType, graph_hyperparameters
from utils import encode_state
from agent import IndependentQLearning
from collections import defaultdict

import time
import create_agents
from reward_functions import final_reward

import matplotlib.pyplot as plt


def _policy(agent_name, agents, observation, done, time_step, episode_num=0):
    """
    Chooses the action for the agent

    agent_name - The agent names
    
    agents - The dictionary of agents

    observations - What the agent can see

    done - Whether the agent is finished

    time_step - The timestep on

    episode_num=0 - The episode number.  Not used

    returns - The action for the agent to run"""

    if time_step > 10:
        return None
    if done:
        return None
    agent = agents[agent_name]
    # print(f'observation: {observation}, and Agent: {agent}')
    return agent.policy(observation, time_step)


def _policyState(agent_name, agents, observation, done, time_step, episode_num=0):
    """
    Chooses the action for the agent

    agent_name - The agent names
    
    agents - The dictionary of agents

    observations - What the agent can see

    done - Whether the agent is finished

    time_step - The timestep on

    episode_num=0 - The episode number.  Not used

    returns - The action for the agent to run"""

    if time_step > 10:
        return None
    if done:
        return None
    agent = agents[agent_name]
    # print(f'observation: {observation}, and Agent: {agent}')
    return agent.play_normal(observation, time_step)


def create_env(num_of_agent, num_of_cycles, local_ratio, multiple, render_mode):
    """
    Creates the environment to be played in

    num_of_agent = 4 - The number of agents to be created.  Default is default for Simple Spread

    num_of_cycles = 10 - The number of frames (step for each agent).  Default is default for Simple Spread


    local_ratio = 0.5 - Weight applied to local and global reward.  Local is collisions global is distance to
    landmarks.  Default is default for simple spread

    multiple = False - Whether you need a parallel environment to be created
    
    render_mode = 'ansi' - Whether to be rendered on screen.  Default is not.
    """
    if multiple:  # decentralised must compute each individually (not parallel)
        return parallel_env(render_mode, num_of_agent, num_of_cycles,
                            local_ratio)
    return env(render_mode="human", N=num_of_agent, max_cycles=num_of_cycles, local_ratio=local_ratio)


centralised = False
NUM_OF_AGENTS = 4
TIMESTEPS_PER_EPISODE = 10
render_mode = 'human'


env = create_env(NUM_OF_AGENTS, TIMESTEPS_PER_EPISODE, 0.5, centralised, render_mode)

# env.reset()

# env.render()

# t = 0

# y = []
# steps = []

# while t<10:
#     steps.append(t)
#     t = t+1
#     actions = {}
#     for agent_name in agents.keys():        # Take action
#         print(f'')
#         agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
#         action = _policy(agent_name, agents, observations[agent_name], False, t-1)
#         actions[agent_name] = action
#     print(actions)

#     observations, rewards, terminations, truncations, infos = env.step(actions)

#     print(f'rewards : {rewards} observation: {observations}')
#     y.append(infos["matrix"].trace())


#     #time.sleep(1)

# print(y)

# plt.plot(steps, y)
# plt.show()

num_episode = int(input('Insert number of episodes: '))

agent_type = AgentType.IQL
agents = {f'agent_{i}': IndependentQLearning(f'agent_{i}', 10) for i in range(NUM_OF_AGENTS)}

adj_table = graph_hyperparameters['graph']
agents = create_marl_agents(NUM_OF_AGENTS, num_episode, NUM_OF_TIMESTEPS,
                            graph_hyperparameters['gamma_hop'], adj_table, graph_hyperparameters['connection_slow'])

agent_old_state = {agent: -1 for agent in agents.keys()}

# env.reset()
# observations = {agent: env.observe(agent) for agent in agents.keys()}


t = 0
rewardX = []
steps = []
for i in range(0, num_episode):

    # print(f'Episode: {i}')
    env.reset()
    observations = {agent: env.observe(agent) for agent in agents.keys()}

    steps.append(i)
    t = 0
    rewardStep = []
    while t < 10:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
            print(observations)
            agent_old_state[agent_name] = encode_state(observations[agent_name], NUM_OF_AGENTS)
            action = _policy(agent_name, agents, observations[agent_name], False, t, num_episode)
            print("we are about to take a step ")
            observations, rewards, terminations, truncations, infos = env.step(action)

        for agent_name in agents.keys():  # Send messages
            agent_obj = agents[agent_name]
            agent_obj.message_passing(num_episode, t, agent_old_state[agent_name], actions[agent_name],
                                      encode_state(observations[agent_name], NUM_OF_AGENTS), rewards[agent_name],
                                      agents)

        for agent_name in agents.keys():  # Update u and v
            agent_obj = agents[agent_name]
            agent_obj.update(num_episode, t, agent_old_state[agent_name],
                             encode_state(observations[agent_name], NUM_OF_AGENTS),
                             actions[agent_name], rewards[agent_name])

        for agent_name in agents.keys():  # Update the values
            agent_obj = agents[agent_name]
            agent_obj.update_values(num_episode, t)

        rewardStep.append(final_reward(rewards))
    while t < 10:
        # steps.append(t)
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
            # agent_old_state[agent_name] = encode_state(observations[agent_name], agent_name)
            agent_old_state[agent_name] = observations[agent_name]
            action = _policy(agent_name, agents, observations[agent_name], False, t - 1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent_name in agents.keys():  # Update the values
            agent_obj = agents[agent_name]
            old_state = agent_old_state[agent_name]
            current_state = observations[agent_name]
            old_action = actions[agent_name]
            reward = rewards[agent_name]
            agent_obj.update_qTable(old_state, current_state, old_action, reward, t - 1)

        rewardStep.append(final_reward(rewards))

        # print(f'printing observation {observations}, reward: {rewards}, infos:{infos}')

    rewardX.append(sum(rewardStep) / t)

    # print(final_reward(rewards))


def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d


for agent in agents:
    print(f'Final Qtable {to_dict(agents[agent].qTable)}')

plt.plot(steps, rewardX)
plt.show()

print("EVALUATING-------")

for i in range(0, 1):

    observations = env.reset()[0]

    t = 0
    while t < 10:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
            agent_old_state[agent_name] = observations[agent_name]
            action = _policyState(agent_name, agents, observations[agent_name], False, t - 1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f'Step: {t} agents actions: {actions.values()}')
        print(f'printing observation {observations}, reward: {rewards}, infos:{infos}')
