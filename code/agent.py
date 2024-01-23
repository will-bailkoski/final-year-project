"""Contains the agent class"""
import random
from collections import defaultdict
from utils import encode_state
from hyperparameters import iql_hyperparameters


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
        greedy = max(self.greedy_value * (self.episode ** -0.5), 0.1)
        #greedy = self.greedy_value
        #print(greedy)

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

