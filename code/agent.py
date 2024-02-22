"""Contains the agent class"""
import random
import math
from collections import defaultdict
from utils import encode_state
from hyperparameters import iql_hyperparameters, ucb_marl_hyperparameters, agent_hyperparameters


class Agent:

    """The basic Agent which plays a random policy"""

    def __init__(self, agent_name):

        """
        Creates a basic Agent

        agent_name - The agent name
        """

        self._agent_name = agent_name

    def policy(self, state, *args):

        """
        Returns a random action

        state - The current state the agent is in

        *args - This is overriden by inherited classes.  None of the arguments are needed

        Returns
        A random move 
        """

        return random.randint(0,1)
    
    def agent_name(self):

        """
        Returns

        The agent name
        """

        return self._agent_name

    def play_normal(self, state, *args):

        """
        This allows the agent to play normally to show on the screen

        state - The state the agent is in

        *args - 

        Returns
        The best move
        """
        
        return self.policy(state)


class IndependentQLearning(Agent):

    """Each Agent learns its own Q-Table"""

    def __init__(self, agent_name, length_of_episode):

        """
        Creates a IQL Agent

        agent_name - The agent name

        length_of_episode - The length of an episode
        """
        self.episode = 0
        self.gamma = iql_hyperparameters['gamma']
        self.greedy_value = iql_hyperparameters['greedy']
        self.alpha = iql_hyperparameters['alpha']
        self.qTable = defaultdict(lambda: defaultdict(lambda: length_of_episode))
        self.H = length_of_episode
        super().__init__(agent_name)

    def getQTable(self):
        return self.qTable

    def policy(self, state, *args):
        """
        Returns an action based off the Q-Table, either random or the best move

        state - The current state the agent is in

        *args - 

        Returns
        An action.
        """        

        # Chooses a random choice with  probability of greedy value
        self.episode += 1
        # greedy = max(self.greedy_value * (self.episode ** -0.5), 0.1)
        greedy = self.greedy_value
        # print(greedy)

        if random.random() < greedy:
            #print('chosing random')

            return super().policy(state, args)
        # Choose the best move
        else:
            #print('chosing according to policy')

            max_value = float('-inf')
            move = -1
            values = [0, 1]
            random.shuffle(values)
            for i in values:
                if self.qTable[state][i] > max_value:
                    max_value = self.qTable[state][i]
                    move = i

            if move == -1:
                #print("Random cos state never seen")
                return super().policy(state)
            else:
                #print("Seen move")
                return move
            

    def play_normal(self, state, *args):  

        """
        This allows the agent to play normally to show on the screen

        state - The state the agent is in

        *args - 

        Returns
        The best action
        """

        # render checks if it is being rendered on the screen

        # This will pick the best action
        max_value = float('-inf')
        move = -1
        values = [0, 1]
        random.shuffle(values)
        for i in values:
            if self.qTable[state][i] > max_value:
                max_value = self.qTable[state][i]
                move = i
        if move == -1:
            return super().policy(state)
        else:

            return move

    def update_qTable(self, old_state, current_state, action, reward, time_step):

        """
        This updates the qTable
        
        old_state - The old state the agent was in

        current_state - The new state the agent is in after taking the action

        action - The action taken from the old_state

        reward - The reward gained

        time_step - The time step we are currently on
        """

        old_value = self.qTable[old_state][action]
        # print(old_value)
        if old_value == float('-inf'):
            old_value = self.H
        old_value_mult = (1-self.alpha) * old_value

        max_value = float('-inf')
        chosen = False
        for i in range(2):  # The actions which can be done
            if self.qTable[current_state][i] > max_value:
                max_value = self.qTable[current_state][i]
                chosen = True
        if not chosen:
            max_value = self.H

        rewards = self.alpha * (reward + (self.gamma*max_value))
        new_value = old_value_mult + rewards
        self.qTable[old_state][action] = new_value


class MARL_Comm(Agent):
    """The original algorithm with communication"""

    def __init__(self, agent_name, num_of_agents, num_of_episodes, length_of_episode, gamma_hop):

        """
        Creates the agent

        agent_name - The name of the agent

        num_of_agents - The number of agents being created

        num_of_episodes - The number of episodes to be trained on

        length_of_episode - The length of one episode

        gamma_hop - The gamma_hop distance for the agent
        """

        # The set of H number of Q-Tables.
        self.qTables = {i + 1: defaultdict(lambda: defaultdict(lambda: length_of_episode)) for i in
                        range(length_of_episode)}

        # This will contain the number of times each state action has been seen for each timestep
        self.nTables = {i + 1: defaultdict(lambda: defaultdict(lambda: 0)) for i in range(length_of_episode)}

        # The u set (episode number is indexed from 0 whilst the timestep is indexed from 1)
        self.uSet = {i: {i + 1: defaultdict(lambda: defaultdict(lambda: set())) for i in
                         range(length_of_episode + gamma_hop + 1)} for i in range(num_of_episodes)}

        # The v set (episode number is indexed from 0 whilst the timestep is indexed from 1)
        self.vSet = {i: {i + 1: defaultdict(lambda: defaultdict(lambda: set())) for i in
                         range(length_of_episode + gamma_hop + 1)} for i in
                     range(-length_of_episode - 1, num_of_episodes + round(gamma_hop % length_of_episode) + 1)}

        self.next_add = set()  # this is for received messages that need to be incorporated into the decision-making

        # Of form agent_name, distance = 0 if no connection
        self._neighbours = {f'agent_{i}': 0 for i in range(num_of_agents)}
        self._num_neighbours = 1
        super().__init__(agent_name)

        self.H = length_of_episode
        self.episodes = num_of_episodes

        # A constant which means bm,t can be solved
        self.c = ucb_marl_hyperparameters['c']

        # This corresponds to A
        num_of_actions = 2

        # This corresponds to S
        num_of_states = agent_hyperparameters['size_of_state_space']

        # This corresponds to T
        T = num_of_episodes * length_of_episode

        # The probablity of failure I believe
        small_prob = ucb_marl_hyperparameters['probability']

        # This is a value used to control the bm,t value.  Actually small iota in paper
        self.l = math.log((num_of_states * num_of_actions * T * num_of_agents) / small_prob)

        # The v-table values.  Set to H as this corresponds to the q tables currently.
        self.vTable = {j + 1: defaultdict(lambda: self.H) for j in range(length_of_episode + 1)}

        self.parity_list = [0] * length_of_episode  # initialised as "no changes have been instructed yet"


    def update_neighbour(self, agent_to_update, connection_quality):

        """
        This updates the connection value in the neighbours dictionary.

        agent_to_update - The agent with a new connection to be updated to

        connection_quality - How long the distance is between agents
        """

        if self._neighbours[agent_to_update] == 0 and connection_quality != 0:
            self._num_neighbours += 1
        elif self._neighbours[agent_to_update] != 0 and connection_quality == 0:
            self._num_neighbours -= 1

        self._neighbours[agent_to_update] = connection_quality

    def policy(self, state, time_step):

        """
        This gets the next move to be played

        state - The current state we are in

        time_step - The time step in the episode

        Return
        The action to be taken
        """

        # Choose the largest value
        q_table = self.qTables[time_step]
        max_value = float('-inf')
        move = -1
        values = [0, 1]
        random.shuffle(values)
        for i in values:
            if q_table[state][i] > max_value:
                max_value = q_table[state][i]
                move = i
        parity_move = (move + self.parity_list[time_step - 1]) % 2
        #print(f"i am {self.agent_name()} on timestep {time_step}. i was originally going to do {move} but now i will do {parity_move}")
        self.parity_list[time_step - 1] = 0
        return parity_move

    def play_normal(self, state, time_step, *args):

        """
        Plays the episode for showing.  Plays the best action in the q-table

        state - The current state we are in

        time_step - The time step in the episode

        *args - Spare arguments.

        Return
        The action to be taken
        """

        q_table = self.qTables[time_step]
        max_value = float('-inf')
        move = -1
        values = [0, 1]
        random.shuffle(values)
        for i in values:
            if q_table[state][i] > max_value:
                max_value = q_table[state][i]
                move = i
        return move

    def choose_smallest_value(self, state, time_step):

        """
        This chooses the smallest value - either the max value from the Q-Table or H

        state - The current state we are in

        time_step - The time step in the episode

        Return
        The smaller value
        """

        max_value = float('-inf')
        values = [0, 1]
        random.shuffle(values)
        for i in values:
            if self.qTables[time_step][state][i] > max_value:
                max_value = self.qTables[time_step][state][i]
        return min(self.H, max_value)

    def message_passing(self, episode_num, time_step, old_state, action, current_state, reward, agents_dict):

        """
        This passes messages to other agents which this agent can communicate to.

        episode_num - The episode number the agent is in

        time_step - The time step the agent is in

        old_state - The state the agent was in

        action - The action taken

        current_state - The new state after the agent has taken

        reward - The reward for the agents move

        agents_dict - The agents dictionary
        """

        for agent in self._neighbours.keys():
            if self._neighbours[agent] != 0:
                agent_obj = agents_dict[agent]  # send a message to every neighbour, receiver is agent_obj
                message_tuple = tuple([time_step, episode_num, self.agent_name(), old_state, action, current_state,
                                       reward])
                agent_obj.receive_message(message_tuple, self._neighbours[agent])

    def update_v_u(self, episode_num, time_step, old_state, current_state, action, reward):

        """
        This updates the u set and v set using the set update rules

        episode_num - The current episode number

        time_step - The time step the agent is on

        old_state - The previous state the agent was on

        current_state - The state the agent is currently on

        action - The action the agent has taken

        reward - The reward gained by the action taken from the old_state
        """

        # This might be how to update the two sets???? Maybe...

        # Update the u-set to contain the latest reward and current state

        # print(episode_num, time_step, old_state, action)
        old_set = self.uSet[episode_num][time_step][old_state][action]
        old_set.add((reward, current_state))
        self.uSet[episode_num][time_step][old_state][action] = old_set

        new_set = set()

        # Add the new data from other agents into vSet.  Only added in the vSet when it 'reaches' the agent (dis == 0).
        # Added into the v-set at the episode and timestep of when the message was sent
        for message_data in self.next_add:
            time_step_other, episode_num_other, agent_name_other, current_state_other, action_other, next_state_other, \
                reward_other, dis = message_data

            if dis == 0:

                old_set = self.vSet[episode_num_other][time_step_other][current_state_other][action_other]
                old_set.add((reward_other, next_state_other))
                self.vSet[episode_num_other][time_step_other][current_state_other][action_other] = old_set
            else:
                dis -= 1
                new_set.add(tuple(
                    [time_step_other, episode_num_other, agent_name_other, current_state_other, action_other,
                     next_state_other, reward_other, dis]))

        self.next_add = new_set

        # Add everything into the vSet for the current episode and timestep
        for state in self.uSet[episode_num][time_step]:
            for act in self.uSet[episode_num][time_step][state]:
                for element in self.uSet[episode_num][time_step][state][act]:
                    reward_new, next_state = element
                    self.vSet[episode_num][time_step][state][action].add((reward_new, next_state))

    def update_q(self, episode_num_max, time_step_max):

        """
        This updates the q_table and the vTable

        episode_num_max - The episode the agent is on

        time_step_max - The time step the agent is on.  Not currently needed
        """

        # This will go over the previous and current episode number in the vSet.  This is because the message should reach the agent within one episode
        for episode_num in range(episode_num_max - 1, episode_num_max + 1):
            # Go over every timestep
            for time_step in range(1, self.H + 1):
                # Go through the vSet
                for state in self.vSet[episode_num][time_step].keys():
                    for action in self.vSet[episode_num][time_step][state].keys():
                        # Updates as specified in the paper
                        for reward, next_state in self.vSet[episode_num][time_step][state][action]:
                            self.nTables[time_step][state][action] = self.nTables[time_step][state][action] + 1

                            t = self.nTables[time_step][state][action]

                            clique_size = self._num_neighbours
                            b = self.c * math.sqrt(((self.H ** 3) * self.l) / (clique_size * t))

                            alpha = (self.H + 1) / (self.H + t)
                            initial = (1 - alpha) * self.qTables[time_step][state][action]
                            expected_future = alpha * (reward + self.vTable[time_step + 1][next_state] + b)
                            new_score = initial + expected_future
                            self.qTables[time_step][state][action] = new_score

                            self.vTable[time_step][next_state] = self.choose_smallest_value(state, time_step)

                # Assume we have gone through all the data in the vSet at this episode and time step.  Means we can add
                # new data (from other agents) but not reuse old data
                self.vSet[episode_num][time_step] = defaultdict(lambda: defaultdict(lambda: set()))

                # print(f"This is {self.agent_name()}'s \n u-set: {self.uSet} \n v-set: {self.vSet} \n q-table: {self.qTables}")

    def receive_message(self, message, dis):

        """
        This is how the agent should receive a message

        message - The tuple containing the message

        dis - How far away the agent sending the message is
        """

        # print(message, dis)

        # message is [time_step, episode_num, self.agent_name, current_state, action, next_state, reward]

        message_list = list(message)
        # print(f"the message passed is {message_list}, received by {self.agent_name()}")
        if len(message_list) == 8:
            self.parity_list[message[0] - 1] = message_list.pop(-1)

        message_list.append(dis)
        self.next_add.add(tuple(message_list))
        # print(f"{self.agent_name()} has messages:")
        # print(self.next_add)

    def message_passing_leader(self, episode_num, time_step, old_state, action, current_state, reward, agents_dict,
                               parity_list):

        if parity_list[0] == 1:
            self.parity_list[time_step - 1] = 1

        # print(f"message passed for episode {episode_num}, timestep {time_step}. parity rn is {parity_list} ")
        for agent in self._neighbours.keys():
            if self._neighbours[agent] != 0:
                agent_obj = agents_dict[agent]  # send a message to every neighbour, receiver is agent_obj
                message_tuple = tuple([time_step, episode_num, self.agent_name(), old_state, action, current_state,
                                       reward, parity_list[list(self._neighbours.keys()).index(agent)]])
                agent_obj.receive_message(message_tuple, self._neighbours[agent])

    def message_passing_revert(self, episode_num, time_step, agents_dict, parity_list):

        if parity_list[0] == 1:
            self.parity_list[time_step - 1] = 1

        # print(f"message passed for episode {episode_num}, timestep {time_step}. parity rn is {parity_list} ")
        for agent in self._neighbours.keys():
            if self._neighbours[agent] != 0:
                agent_obj = agents_dict[agent]  # send a message to every neighbour, receiver is agent_obj
                message_tuple = tuple([time_step, episode_num, self.agent_name(), parity_list[list(self._neighbours.keys()).index(agent)]])
                agent_obj.receive_message(message_tuple, self._neighbours[agent])

    def printlist(self):
        return self.parity_list
