"""This is a simulation of agents acting in a decentralised framework with no message passing"""
from pettingZoo.PettingZoo.pettingzoo.mpe.custom_env.custom_env import env, parallel_env
from hyperparameters import train_hyperparameters, agent_hyperparameters, AgentType
from agent import IndependentQLearning
from collections import defaultdict

import time
from reward_functions import final_reward

import matplotlib.pyplot as plt


def _policy(agent_name, agents, observation, done, time_step):
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
    # print("we are here")
    return agent.policy(observation, time_step)


def _policyState(agent_name, agents, observation, done, time_step):
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
    return agent.play_normal(observation, time_step)


multiple = True
NUM_OF_AGENTS = 4
NUM_OF_TIMESTEPS = 10

if multiple:
    env = parallel_env(render_mode="human", N=NUM_OF_AGENTS, max_cycles=NUM_OF_TIMESTEPS, local_ratio=0.5)
else:
    env = env(render_mode="human", N=NUM_OF_AGENTS, max_cycles=NUM_OF_TIMESTEPS, local_ratio=0.5)

NUM_OF_EPISODES = int(input('Insert number of episodes: '))
agents = {f'agent_{i}': IndependentQLearning(f'agent_{i}', NUM_OF_TIMESTEPS) for i in range(NUM_OF_AGENTS)}

rewardX = []
steps = []
for i in range(0, NUM_OF_EPISODES):

    if (i % 50) == 0:
        print(i)

    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset()[0]
    steps.append(i)
    t = 0
    rewardStep = []
    while t < NUM_OF_TIMESTEPS:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
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

    # EVALUATION

    observations = env.reset()[0]
    t = 0
    while t < NUM_OF_TIMESTEPS:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():
            action = _policyState(agent_name, agents, observations[agent_name], False, t - 1)
            actions[agent_name] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        rewardStep.append(final_reward(rewards))

    rewardX.append(sum(rewardStep) / t)

def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d


for agent in agents:
    print(f'Final Qtable {to_dict(agents[agent].qTable)}')

plt.plot(steps, rewardX)
plt.show()

print("EVALUATING-------")

repeats = 1
t_opt_policy = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
for i in range(0, repeats):

    observations = env.reset()[0]
    print(f"Initial observations: {observations}")

    t = 0
    total = []
    while t < 10:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
            agent_old_state[agent_name] = observations[agent_name]
            action = _policyState(agent_name, agents, observations[agent_name], False, t - 1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f'Step: {t} agents actions: {actions.values()}')
        total.append(sum(actions.values()))
        print(f'printing observation {observations}, reward: {rewards}, infos:{infos}')
        time.sleep(0.5)

    print(f'The total number of agents processing on each time step was {total}')
    assert(
            total == t_opt_policy
    ), "this did not match the optimal policy"
