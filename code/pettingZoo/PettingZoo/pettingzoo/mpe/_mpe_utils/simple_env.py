import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

import traceback

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        test=False,
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self.seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random, test)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector = agent_selector(self.agents)
        
        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            #obs 
            #obs_dim = len(self.scenario.observation(agent, self.world))

            #print(self.scenario.observation(agent, self.world))


            # state_dim += obs_dim
            '''
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(
                    low=0, high=1, shape=(space_dim,)
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)

             '''  
            self.action_spaces[agent.name] = spaces.Discrete(2) 
            # self.observation_spaces[agent.name] = spaces.Box(
            #     low=-np.float32(2),
            #     high=+np.float32(2),
            #     shape=(obs_dim,),
            #     dtype=np.float32,
            # )

        #print(state_dim)
        self.state_space = spaces.Discrete(10)
         

        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        #print( f'{self.scenario.observation(self.world.agents[self._index_map[agent]], self.world)}' )
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def state(self):
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random, test=options)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    def _execute_world_step(self):
        # set action for each agent

        

        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            #print(action)
            #print(agent.movable)
            scenario_action = []
            if agent.processing: #adding the reset
                scenario_action.append(action)
                if action==0:
                    #print('setting sensor toraw')
                    agent.sensor.set_raw()
                    agent.sensor.reset()

                elif action==1:
                    #print('setting sensor to proc')
                    agent.sensor.set_proc()
                    agent.sensor.reset()
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            #print(scenario_action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.scenario.avg_err_var = 0
        for t in range(self.scenario.window_length):

            # update delays
            self.scenario.stored_delays = [delay + 1 for delay in self.scenario.stored_delays]





            # store available data
            sensor_has_new_data = False
            for agent in self.world.agents:

                agent.sensor.time_step()

                new_data_from_sensor = self.scenario._collect_data(agent.sensor, t)
                if not(sensor_has_new_data) and new_data_from_sensor:
                    sensor_has_new_data = True
            
            
            # update current error covariance
            if sensor_has_new_data:
                self.scenario.P = self.scenario._open_loop(self.scenario.stored_cov[-1], self.scenario.stored_delays[-1])

            else:
                self.scenario.P = self.scenario._open_loop(self.scenario.P, 1)

            #print(f'Sensor has new data {sensor_has_new_data}, {self.scenario.P}')

            # update average error variance
            err_var_curr = np.trace(self.scenario.P)
            self.scenario.avg_err_var += (err_var_curr - self.scenario.avg_err_var) / (t+1)
            self.scenario.Ps.append(err_var_curr)

        #print(f'Ps in execute step world {err_var_curr}')

        self.infos["matrix"]=self.scenario.P

        self.scenario.timer+= 1
        '''†
        # update processing sensors
        if proc_sens_new < self.proc_sens:

            # set sensors to raw mode
            for to_raw in range(proc_sens_new, self.proc_sens):
                self.sensors[to_raw].set_raw()
                self.sensors[to_raw].reset()

        elif proc_sens_new > self.proc_sens:

            # set sensors to processing mode
            for to_proc in range(self.proc_sens, proc_sens_new):
                self.sensors[to_proc].set_proc()
                self.sensors[to_proc].reset()

        # update number of sensors in processing mode
        self.proc_sens = proc_sens_new
        '''


        


        # update sensors
        # compute reward for this episode (time window)
        #state, reward = self._compute_reward()
        #state, reward = self.world.step()

        #WE SHOULD HAVE STATE AND REWARD

        self.world.step()
        # check condition for episode termination
        done = False
        if self.scenario.timer == self.scenario.window_num:
            done = True

       
        
        #print('first than exec')


        global_reward = 0.0
        # if self.local_ratio is not None:
        #     global_reward = float(self.scenario.global_reward(self.world))
        
        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward
        
        #print('And end here')
         # we shall return this things return state, reward, done

        

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        #print(f'action : {action} agent: {agent}')
                 
        if agent.processing:
            if not self.continuous_actions:
                if action[0] == 0:
                    agent.action.u[0]=0
                    agent.color = np.array([0.35, 0.35, 0.85])
                if action[0] == 1:
                    agent.action.u[0]=1
                    agent.color = np.array([0.75, 0.75, 0.75])
            action = action[1:]

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
           # print(f'self.continuous_actions {self.continuous_actions}')
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        # print("we are on line 361 in simple_env, action is ", str(action))
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action
        # print("next_id is ", next_idx)
        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.WARN(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            self.draw()
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
