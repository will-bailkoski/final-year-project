"""This is a simulation of agents acting in a decentralised framework with """
from pettingZoo.PettingZoo.pettingzoo.mpe.custom_env.custom_env import env, parallel_env
from agent import IndependentQLearningWithLeader
from collections import defaultdict
import random
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
agents = {f'agent_{i}': IndependentQLearningWithLeader(f'agent_{i}', NUM_OF_TIMESTEPS) for i in range(NUM_OF_AGENTS)}

leader = list(agents.keys())[0]
parity_list = ([0] * (NUM_OF_AGENTS - 1)) + [1]
max_explore = {i + 1: ({agent: None for agent in agents.keys()}, float(-1000000)) for i in range(NUM_OF_TIMESTEPS)}
opt_dict = {1 : {'agent_0': 0, 'agent_1': 0, 'agent_2': 1, 'agent_3': 0}, 2 : {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, 3: {'agent_0': 1, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, 4: {'agent_0': 0, 'agent_1': 1, 'agent_2': 0, 'agent_3': 0}, 5: {'agent_0': 0, 'agent_1': 0, 'agent_2': 1, 'agent_3': 0}, 6:{'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 1}, 7:{'agent_0': 0, 'agent_1': 0, 'agent_2': 1, 'agent_3': 0}, 8:{'agent_0': 0, 'agent_1': 1, 'agent_2': 0, 'agent_3': 0}, 9:{'agent_0': 1, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, 10:{'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 1}}
test = [{'agent_0': 1, 'agent_1': 0, 'agent_2': 1, 'agent_3': 1}, {'agent_0': 0, 'agent_1': 1, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 1, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 1, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 1}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}, {'agent_0': 0, 'agent_1': 0, 'agent_2': 0, 'agent_3': 0}]
avg_for_ep = []
rewardX = []
steps = []
ep_best = float("-inf")
for i in range(0, NUM_OF_EPISODES):

    if (i % 50) == 0:
        print(i)

    print(f"Episode {i}:")

    agent_old_state = {agent: -1 for agent in agents.keys()}
    observations = env.reset()[0]
    steps.append(i)
    t = 0
    rewardStep = []
    while t < NUM_OF_TIMESTEPS:
        t = t + 1
        actions = {}
        n = 0
        for agent_name in agents.keys():  # Take action
            # agent_old_state[agent_name] = encode_state(observations[agent_name], agent_name)
            # print(f' {observations} type: {type(observations)}  0 element {observations[0]} type {type(observations[0])} agent name: {agent_name}')
            agent_old_state[agent_name] = observations[agent_name]
            action = _policy(agent_name, agents, observations[agent_name], False, t - 1)  # choose action
            actions[agent_name] = action
        # actions = opt_dict[t]

        observations, rewards, terminations, truncations, infos = env.step(actions)  # execute actions
        #  print(f"ep {i}, time {t}, actions taken are {actions}")

        step_reward = -1 * infos['matrix'].trace()  # final_reward(rewards)
        if step_reward > max_explore[t][1]:# "is this set of actions better than our previous best?"
            max_explore[t] = (actions, step_reward)
            print("NEW BEST!")
            print(t, max_explore[t])
            agents[leader].message_passing_leader(t, agents, max_explore[t][0], [0, 0, 0, 0])  # Confirm it's best
        else:
            random.shuffle(parity_list)
            agents[leader].message_passing_leader(t, agents, max_explore[t][0], parity_list)  # Try something new

        for agent_name in agents.keys():  # Update the values
            agent_obj = agents[agent_name]
            old_state = t  # agent_old_state[agent_name]
            current_state = t + 1  # observations[agent_name]
            old_action = actions[agent_name]  # max_explore[t][0][agent_name]
            reward = step_reward  # max_explore[t][1]  # rewards[agent_name]
            agent_obj.update_qTable(old_state, current_state, old_action, reward, t -1)

    # EVALUATION

    observations = env.reset()[0]
    t = 0
    actionstaken = []
    while t < NUM_OF_TIMESTEPS:
        t = t + 1
        actions = {}
        for agent_name in agents.keys():
            # action = _policyState(agent_name, agents, observations[agent_name], False, t - 1)
            action = _policyState(agent_name, agents, t, False, t - 1)
            actions[agent_name] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        actionstaken.append(actions)

        rewardStep.append(-1 * infos['matrix'].trace())

    ep_rew = sum(rewardStep) / 10
    print(ep_rew)
    if ep_rew > ep_best:
        ep_best = ep_rew
        print("THE BEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEST!!!!!")
        print(actionstaken)

    rewardX.append(sum(rewardStep) / 10)

print(avg_for_ep)
print(sum(avg_for_ep) / 10)

sums = 0
for x in max_explore:
    sums += max_explore[x][1]



def to_dict(d):
    if isinstance(d, defaultdict):
        return dict((k, to_dict(v)) for k, v in d.items())
    return d


for agent in agents:
    print(f'Final Qtable {to_dict(agents[agent].qTable)}')

# print(steps)
# print(rewardX)

plt.plot(steps, rewardX)
plt.show()

print("EVALUATING-------")
print(max_explore)

sumsl = 0

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
            action = _policyState(agent_name, agents, t, False, t - 1)
            actions[agent_name] = action
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f'Step: {t} agents actions: {actions.values()}')
        total.append(sum(actions.values()))
        print(f'printing observation {observations}, reward: {rewards}, infos:{infos}')
        sumsl += -1 * infos['matrix'].trace()
        time.sleep(0.5)

    print(f'The total number of agents processing on each time step was {total}')
    print(f"reward for this policy is {sumsl / 10}")

    # assert(
    #         total == t_opt_policy
    # ), "this did not match the optimal policy"

