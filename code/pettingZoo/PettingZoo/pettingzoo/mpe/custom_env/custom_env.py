import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.utils.conversions import parallel_wrapper_fn

from .sensor import Sensor

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env

from numpy.core.fromnumeric import size

from typing import Tuple, List


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N=4,
            local_ratio=0.5,
            max_cycles=10,
            continuous_actions=False,
            render_mode=None,
    ):
        EzPickle.__init__(
            self, N, local_ratio, max_cycles, continuous_actions, render_mode
        )
        assert (
                0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        # for i, lm in enumerate(world.landmarks):
        #     print(lm.state.p_pos)
        #     lm.state.p_pos = np.array([0.5+ i, 0.5 + i])
        #     print(lm.state.p_pos)

        # for i, agent in enumerate(world.agents):
        #     print(agent.state.p_pos)
        #     agent.state.p_pos = np.array([1 + i, 1 + i])
        #     print(agent.state.p_pos)

        self.metadata["name"] = "custom_env_v1"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, N=4):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = 1
        world.collaborative = True
        # add agents

        self.state_bins = []
        self.state_dim = 0  # state space dimension
        self.state = None

        # System dynamics
        T = .01  # sampling time
        A = np.array([[1, T], [0, 1]])  # state dynamics
        Q = 1e3 * np.array([[0, 0], [0, T ** 2]])  # covariance of process noise
        # Smart sensors
        N = 4  # amount of sensors
        C = np.array([1, 0])  # state-to-output transformation
        C = np.transpose(C[:, np.newaxis])
        V_raw = 10  # covariance of measurement noise of raw data
        V_proc = 1  # covariance of measurement noise of processed data
        del_raw = int(100 / 25)  # computation delay of raw measurements
        del_proc = del_raw + 10  # computation delay of processed measurements
        comm_del_raw = 1  # communication delay of raw measurements
        comm_del_proc = 1  # communication delay of processed measurements
        enabled_at_start = False  # sensor starts in processing mode
        self.A = A.copy()  # state matrix
        self.Q = Q.copy()  # stored_covariance of process noise
        self.n = size(A, 1)  # dynamical state dimension
        self.P0 = 10 * np.identity(self.n)  # error covarance initial condition
        self.stored_cov = None  # stored error covariances
        self.P = None  # current error covariance
        self.Ps = []  # error covariance history

        self.rrr = 0
        # parameters for sensor of the parameter, fixed for everyone at the beginnig
        V_raw = 10  # covariance of measurement noise of raw data
        V_proc = 1  # covariance of measurement noise of processed data
        del_raw = int(100 / 25)  # computation delay of raw measurements
        del_proc = del_raw + 10  # computation delay of processed measurements
        comm_del_raw = 1  # communication delay of raw measurements
        comm_del_proc = 1  # communication delay of processed measurements
        enabled_at_start = False  # sensor starts in processing mode
        C = np.array([1, 0])  # state-to-output transformation
        C = np.transpose(C[:, np.newaxis])  # Error Covariance matrix
        # self.P = np.zeros((len(self.world.agents), len(self.world.agents)))
        self.P = None
        self.rho = 0

        # parameters
        # Time horizon for optimization
        time_horizon = int(5e2)
        window_length = int(5e1)  # decision interval
        window_num = int(time_horizon // window_length)  # time horizon = window_num * window_length

        # processing
        # bins for state space quantization
        self.proc_sens = 0  # amount of sensors currently processing

        self.window_length = int(window_length)  # length of time window
        self.window_num = window_num  # number windows such that K = window_length * window_num
        self.timer = 0  # tracks number of windows run so far
        self.STORAGE_DIM = 100  # amount of measurements that can be  at the central station
        self.stored_delays = None  # delays of stored measurements
        self.stored_info_mat = None  # information matrices of stored measurements

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.movable = False
            agent.size = 0.05
            agent.sensor = Sensor(
                del_raw, V_raw, del_proc, V_proc, enabled_at_start,
                C, comm_del_raw, comm_del_proc)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = True
            landmark.silent = True
        return world

    def reset_world(self, world, np_random, test=False):

        bins = [6.6, 7.3, 9.0, 12.5]
        self.set_quantization(bins)

        # added processing network-----

        self.stored_delays = [np.inf for _ in range(self.STORAGE_DIM)]
        self.stored_info_mat = [np.zeros((self.n, self.n)) for _ in range(self.STORAGE_DIM)]
        self.stored_cov = [self.P0.copy() for _ in range(self.STORAGE_DIM)]
        self.P = self.P0.copy()

        # print('setting random sensors')  #why sensor are random setted to proc and to raw

        for agent in world.agents:

            if np.random.randint(0, 1) == 0:

                agent.sensor.set_proc()
                agent.sensor.reset()
            else:
                agent.sensor.set_raw()
                agent.sensor.reset()

        self.timer = 0
        self.Ps = []
        self.state = self._quantize(np.trace(self.P0))

        # print(f'initial state {self.state}')

        # ----

        # print(world.P)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states.  These are the continuous values.

        for i, agent in enumerate(world.agents):
            if i == 0:
                # If test is true we swap A & D
                if test:
                    agent.state.p_pos = np.array([1 / 1.414, 1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                    agent.state.p_vel = np.array(
                        [1 / 1.414, 1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
                else:
                    agent.state.p_pos = np.array([-1 / 1.414, 1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                    agent.state.p_vel = np.array(
                        [-1 / 1.414, 1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
            if i == 1:
                agent.state.p_pos = np.array([-1 / 1.414, -1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array(
                    [-1 / 1.414, -1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
            if i == 2:
                agent.state.p_pos = np.array([1 / 1.414, -1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.array(
                    [1 / 1.414, -1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
            if i == 3:
                # If test is true we swap A & D
                if test:
                    agent.state.p_pos = np.array([-1 / 1.414, 1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                    agent.state.p_vel = np.array(
                        [-1 / 1.414, 1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)
                else:
                    agent.state.p_pos = np.array([1 / 1.414, 1 / 1.414])  # np_random.uniform(-1, +1, world.dim_p)
                    agent.state.p_vel = np.array(
                        [1 / 1.414, 1 / 1.414])  # agent.state.p_vel = np_random.uniform(-1, +1, world.dim_p)

            # agent.state.p_vel = np.array([0,0])
            # These are the observed values - related to the continuous values but clipped to be in the state space
            agent.state.obs_pos = np.zeros(world.dim_p)
            agent.state.obs_vel = np.zeros(world.dim_p)
            self.convert_values(agent)
            agent.state.c = np.zeros(world.dim_c)

            agent.state.realState = 0

        # position = np_random.uniform(-2, +2, world.dim_p)
        # position = np.array([0 for num in range(world.dim_p)])     # So all surrounding the origin.
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np_random.uniform(-2, +2, world.dim_p)
            # landmark.state.p_pos = position
            landmark.state.p_pos = [-1 / 1.414, 0]
            landmark.state.p_vel = np.zeros(world.dim_p)

            landmark.state.obs_pos = np.zeros(world.dim_p)
            landmark.state.obs_vel = np.zeros(world.dim_p)

            self.convert_values(landmark)

    def set_quantization(self, bins: List[float]):
        '''
        Set state space quantization for the environment.
        
        Input
        -----
        bins: list[float],
            thresholds of bins thaqt define the state space of the environment         
        '''
        self.state_bins = bins
        self.state_dim = len(bins) + 1

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def _collect_data(self, sensor: Sensor, t: int) -> bool:
        '''
        Collects sensory data and updates state estimate and estimation error covariance matrix.
        
        Input
        -----
        sensor: Sensor,
            smart sensor in the network
        t: int,
            current time
            
        Returns
        -------
        new_data_collected: bool,
            True if new data have been transmitted from sensor
        '''
        new_data_collected = False
        if sensor.has_new_data():
            delay, info_mat = sensor.new_data()
            if delay < self.stored_delays[0]:
                new_data_collected = True
                new_data_stored = False

                # store new measurement and remove oldest one (one mesurement per each memory cell)
                for index, stored_delay in enumerate(self.stored_delays):

                    # if new measurement already stored, update error covariances associated with fresher measurements
                    if new_data_stored:
                        P_temp = self._open_loop(self.stored_cov[index - 1],
                                                 self.stored_delays[index - 1] - stored_delay)
                        self.stored_cov[index] = self._update(P_temp, self.stored_info_mat[index])

                    else:
                        if delay >= stored_delay:

                            # update outdated error covariance associated with new measurement
                            if self.stored_delays[index - 1] < np.inf:
                                P_temp = self._open_loop(self.stored_cov[index - 1],
                                                         self.stored_delays[index - 1] - delay)

                            else:

                                # fill array at the beginning
                                P_temp = self._open_loop(self.stored_cov[index - 1], t - delay)

                            self.stored_cov[index - 1] = self._update(P_temp, info_mat)

                            # store new measurement
                            self.stored_delays[index - 1] = delay
                            self.stored_info_mat[index - 1] = info_mat.copy()
                            new_data_stored = True

                            # update outdated error covariance associated with currently iterated measurement
                            P_temp = self._open_loop(self.stored_cov[index - 1], delay - stored_delay)
                            self.stored_cov[index] = self._update(P_temp, self.stored_info_mat[index])

                        else:
                            if index > 0:
                                # overwrite old data with fresher ones
                                self.stored_delays[index - 1] = stored_delay
                                self.stored_info_mat[index - 1] = self.stored_info_mat[index].copy()
                                self.stored_cov[index - 1] = self.stored_cov[index].copy()

                if not (new_data_stored):
                    if self.stored_delays[-1] < np.inf:
                        P_temp = self._open_loop(self.stored_cov[-1], self.stored_delays[-1] - delay)

                    else:

                        # fill array at the beginning (first measurement)
                        P_temp = self._open_loop(self.stored_cov[-1], t - delay)

                    self.stored_cov[-1] = self._update(P_temp, info_mat)
                    self.stored_delays[-1] = delay
                    self.stored_info_mat[-1] = info_mat.copy()

        return new_data_collected

    def _update(self, P0: np.ndarray, info_mat: np.ndarray) -> np.ndarray:
        '''
        Runs update with measurement of Kalman filter.
        
        Input
        -----
        P0: numpy array,
            initial estimation error covariance matrix
        info_mat: numpy array,
            information matrix associated with measurement
            
        Returns
        -------
        P_update: numpy array,
            estimation error covariance matrix after update with measurement
        '''
        if self.n > 1:
            P_update = np.linalg.inv(np.linalg.inv(P0) + info_mat)

        else:
            P_update = (P0 ** -1 + info_mat) ** -1

        return P_update

    def _open_loop(self, P0: np.ndarray, T: int) -> np.ndarray:
        '''
        Runs open loop step of Kalman filter.
        
        Input
        -----
        P0: numpy array,
            initial estimation error covariance matrix
        T: int,
            numer of open-loop steps
            
        Returns
        -------
        P_t: numpy array,
            estimation error covariance matrix after T open-loop steps
        '''
        if T == 0:
            P_T = P0

        else:
            Q_tot = self.Q.copy()
            for t in range(1, T):
                A_t = np.linalg.matrix_power(self.A, t)
                Q_tot += np.matmul(A_t, np.matmul(self.Q, np.transpose(A_t)))

            A_T = np.linalg.matrix_power(self.A, T)
            P_T = np.matmul(A_T, np.matmul(P0, np.transpose(A_T))) + Q_tot

        return P_T

    def _quantize(self, err_var: float) -> int:
        '''
        Find quantization bin corresponding to the trace of the estimation error covariance matrix.
        
        Input
        -----
        err_var: float,
            trace of the estimation error covariance matrix
            
        Returns
        -------
        bin: int,
            quantization bin corresponding to err_var
        '''
        for i in range(self.state_dim - 1):
            if err_var < self.state_bins[i]:
                return i

        return self.state_dim - 1

    def reward(self, agent, world):
        # Agents rewarded for how close they are to their landmark
        # print("I reach here")

        # is the compute reward return 2 variable

        # print(world.P)

        # print(f'Im here')

        # print(f'Ps in reward: {self.Ps[-1]}')

        self.state = self._quantize(self.Ps[-1])
        reward = -self.avg_err_var

        # print(f'REWAAARS state: {self.state} reward {reward}')

        self.rrr = reward

        return reward

        # original return state and error

        # return -np.trace(world.P*world.rho)
        # for lm in world.landmarks:
        #     print('aaa',lm.state.p_pos)

        #     if lm.name[-1] == agent.name[-1]:
        #         self.convert_values(agent)
        #         self.convert_values(lm)
        #         # print(lm.state.p_pos)
        #         return -np.sqrt(np.sum(np.square(agent.state.obs_pos - lm.state.obs_pos)))

    def observation(self, agent, world):
        # # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:  # world.entities:
        #     entity_color.append(entity.color)
        # # communication of all other agents
        # comm = []
        # other_pos = []
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)

        # self.convert_values(agent)

        # print(f'real observation: {np.asarray(agent.state.realState)}')
        # print(np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos]
        # ))
        # return agent.state.realState

        # print(f'Printig state {self.state}')

        # print(f'STAAATE state: {self.state} reward {self.rrr}')

        return self.state

    def convert_values(self, entity):
        # print("HELLO")
        # correct_values = np.array([-2, -1.56, -1.11, -0.67, -0.22, 
        #                 0.22, 0.67, 1.11, 1.56, 2])

        # These are the 10 values which we can match to.
        correct_values = np.array([-2, -1.6, -1.2, -0.8, -0.4,
                                   0.0, 0.4, 0.8, 1.2, 1.6])

        # This converts the position value
        for i, value in enumerate(entity.state.p_pos):
            smallest_distance = abs(correct_values - value)  # Something like this
            index_of_smallest = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest))
            # print(value, correct_values)
            # print(index_of_smallest)
            # print(index_of_smallest[0])
            if len(index_of_smallest[0]) > 1:
                index_of_smallest = index_of_smallest[0][0]
            # print(index_of_smallest)
            entity.state.obs_pos[i] = correct_values[index_of_smallest]

        # This converts the velocity value
        for i, value in enumerate(entity.state.p_vel):
            smallest_distance = abs(correct_values - value)  # Something like this
            index_of_smallest_p_vel = np.where(smallest_distance == min(smallest_distance))
            # print(type(index_of_smallest_p_vel))
            # print(value, correct_values)
            # print(index_of_smallest_p_vel)
            # print(index_of_smallest_p_vel[0])
            if len(index_of_smallest_p_vel[0]) > 1:
                index_of_smallest_p_vel = index_of_smallest_p_vel[0][0]

            entity.state.obs_vel[i] = correct_values[index_of_smallest_p_vel]
