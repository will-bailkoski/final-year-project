import hashlib
import numpy as np

# Taken from the MARL work please cite for me FUTURE NAT
def encode_state(observation, num_of_agents):

    """
    Encodes a state which can be saved over Python instances.  The hash function changes after every Python instance
    Taken from *Gertjan Verhoeven* Notebook found on PettingZoo website (Which has been subsequently deleted) 
    
    observation - What the agent can see

    num_of_agents - The number of agents overall. Unneeded variable

    returns - an encoded state"""

    # encode observation as bytes
    # print(f'agent observation {observation}')      
         
    # print(num_of_agents)
    #print(str(observation)+str(num_of_agents))
    obs_bytes = (str(observation)).encode('utf-8')
    # create md5 hash
    m = hashlib.md5(obs_bytes)
    # return hash as hex digest
    state = m.hexdigest()
    return(state)