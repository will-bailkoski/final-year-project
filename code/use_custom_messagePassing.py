"""This will create the necessary environment for use of the rest of the program.  Plug & play with this
abstracting which environment is used"""
from pettingZoo.PettingZoo.pettingzoo.mpe.custom_env.custom_env import env, parallel_env
from create_agents import create_marl_agents
from hyperparameters import train_hyperparameters, agent_hyperparameters, AgentType, graph_hyperparameters
from collections import defaultdict

import random
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
    return agent.play_normal(observation, time_step)


multiple = True
NUM_OF_AGENTS = 4
NUM_OF_TIMESTEPS = 10

if multiple:
    env = parallel_env(render_mode="human", N=NUM_OF_AGENTS, max_cycles=NUM_OF_TIMESTEPS, local_ratio=0.5)
else:
    env = env(render_mode="human", N=NUM_OF_AGENTS, max_cycles=NUM_OF_TIMESTEPS, local_ratio=0.5)

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

NUM_OF_EPISODES = int(input('Insert number of episodes: '))


agents = create_marl_agents(NUM_OF_AGENTS, NUM_OF_EPISODES, NUM_OF_TIMESTEPS,
                            graph_hyperparameters['gamma_hop'],
                            graph_hyperparameters['graph'],
                            graph_hyperparameters['connection_slow'])

agent_old_state = {agent: -1 for agent in agents.keys()}

t = 0
rewardX = []
rewardsimprov = []
steps = []

max_explore = {i + 1: (None, float("-inf")) for i in range(NUM_OF_TIMESTEPS)}
leader = list(agents.keys())[0]
explorers = list(agents.keys())[1:]
parity_list = ([0] * (NUM_OF_AGENTS - 1)) + [1]

for i in range(0, NUM_OF_EPISODES):

    if (i % 50) == 0:
        print("episode: ", i)

    observations = env.reset()[0]
    # observations = {agent: env.observe(agent) for agent in agents.keys()}
    steps.append(i)
    t = 0
    rewardStep = []
    max_reward = []
    #
    # for agent_name in agents.keys():
    #     agent_obj = agents[agent_name]
    #     print(f"{agent_obj.agent_name()} has parity list {agent_obj.printlist()}")

    while t < NUM_OF_TIMESTEPS:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Decide and take action
            agent_obj = agents[agent_name]
            agent_old_state[agent_name] = observations[agent_name]
            action = _policy(agent_name, agents, observations[agent_name], False, t)  # decide action
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)  # take actions
        # print(f"actions taken are {list(actions.values())}")

        step_reward = final_reward(rewards)
        new_best = False
        if step_reward > max_explore[t][1]:  # "is this set of actions better than our previous best?"
            max_explore[t] = (actions, step_reward)
            new_best = True  # "then we don't need to worry about changing our actions this episode"

        old_parity = parity_list[:]
        random.shuffle(parity_list)
        parity_message = parity_list[:]
        if not new_best:
            for j in range(len(parity_list)):
                parity_message[j] = (old_parity[j] + parity_list[j]) % 2

        # print(f"{new_best} leader is sending {parity_message}")

        agent_obj = agents[leader]
        agent_obj.message_passing_leader(i,  # i or NUM_OF_EPISODES
                                             t, agent_old_state[leader], actions[leader],
                                             observations[leader], rewards[leader],
                                             agents, parity_message)

        for agent_name in explorers:  # Send messages
            agent_obj = agents[agent_name]
            agent_obj.message_passing(i,  # i or NUM_OF_EPISODES
                                      t, agent_old_state[agent_name], actions[agent_name],
                                      observations[agent_name], rewards[agent_name],
                                      agents)

        for agent_name in agents.keys():  # Update u and v
            agent_obj = agents[agent_name]
            agent_obj.update_v_u(i, t, agent_old_state[agent_name],
                                 observations[agent_name], actions[agent_name], rewards[agent_name])

        for agent_name in agents.keys():  # Update the values
            agent_obj = agents[agent_name]
            agent_obj.update_q(i, t)

        # for agent_name in agents.keys():  # Update the values
        #     agent_obj = agents[agent_name]
        #     old_state = agent_old_state[agent_name]
        #     current_state = observations[agent_name]
        #     old_action = actions[agent_name]
        #     reward = rewards[agent_name]
        #     agent_obj.update_qTable(old_state, current_state, old_action, reward, t - 1)

        rewardStep.append(step_reward)
        max_reward.append(max_explore[t][1])

    rewardsimprov.append(sum(max_reward) / NUM_OF_TIMESTEPS)
    rewardX.append(sum(rewardStep) / NUM_OF_TIMESTEPS)

print(max_explore)

def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d


for agent in agents:
    print(f'Final Qtable {to_dict(agents[agent].qTables)}')

plt.plot(steps, rewardX)
plt.show()

plt.plot(steps, rewardsimprov)
plt.show()

print("EVALUATING-------")

for i in range(0, 1):

    observations = env.reset()[0]

    t = 0
    total = []
    while t < 10:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():  # Take action
            agent_old_state[agent_name] = observations[agent_name]
            action = _policyState(agent_name, agents, observations[agent_name], False, t)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f'Step: {t} agents actions: {actions.values()}')
        total.append(sum(actions.values()))
        print(f'printing observation {observations}, reward: {rewards}, infos:{infos}')

    print(f'The total number of agents processing on each time step was {total}')