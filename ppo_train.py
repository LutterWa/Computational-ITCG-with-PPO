import random
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from OAgent_sim import OAgentSim
from ppo_model import PPO
from dnn_test import DeepNeuralNetwork

BATCH_SIZE = 256  # update batch size
TRAIN_EPISODES = 1000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
GAMMA = 0.995
REWARD_SAVE_CASE = 0
dnn = DeepNeuralNetwork()


class OMissile(OAgentSim):
    def __init__(self, no_=0, missile=None, target=None):
        super().__init__(no_=0, missile=None, target=None)
        self.tgo = float(dnn.predict([[self.Y[1] / 315, self.Y[2] / -0.6, self.Y[3] / -9.3e3, self.Y[4] / 1.3e4]]))

    def get_tgo(self, dnn_state=None):
        if dnn_state is None:
            dnn_state = [self.Y[1] / 315, self.Y[2] / -0.6, self.Y[3] / -9.3e3, self.Y[4] / 1.3e4]
        self.tgo = tgo = float(dnn.predict([dnn_state]))
        return tgo

    def get_state(self, t_target):
        v, theta, r, q, x, y, t = self.collect()
        tgo = self.get_tgo([v / 315., theta / -0.6, x / -9.3e3, y / 1.3e4])
        state_local = [v / 315.,  # 0.speed
                       theta / -0.6,  # 1.the angle of v
                       # r / 1.6e4,  # 2.range
                       # -q,  # 3.line of sight angle
                       x / -9.3e4,  # 4. x
                       y / 1.3e4,  # 5.heights of missile
                       max((t_target - (t + tgo)) / 2, 0)]  # 6.time error
        return np.array(state_local)

    def get_reward(self, t_target):
        e_local = (t_target - (self.Y[0] + self.tgo)) / self.tgo
        ky = 0.1
        reward_local = ky * (0.1 * math.exp((self.Y[4] - self.R) / 1.3e4) +
                             0.9 * math.exp(-(self.R / 1.6e4) ** 2)) + \
                       (1 - ky) * math.exp(-e_local ** 2)
        # print(zem)
        return np.array(reward_local)


if __name__ == '__main__':
    env = OMissile()

    # set the init parameter
    state_dim = 5
    action_dim = 1
    action_bound = 3 * 9.81  # action limitation
    t0 = time.time()
    model_num = 0

    train = False  # choose train or test
    if train:
        agent = PPO(state_dim, action_dim, action_bound)
        dict_episode_reward = {'all_episode_reward': [], 'episode_reward': [], 'target_time': [], 'actual_time': []}
        dict_episode_time = {'desired tgo': [], 'actual tgo': [], 'impact time error': []}
        all_episode_reward = []
        for episode in range(int(TRAIN_EPISODES)):
            env.modify()  # 0.时间,1.速度,2.弹道倾角,3.导弹x,4.导弹y,5.质量
            td = env.get_tgo() * random.uniform(1.1, 1.2)
            desired_tgo = []  # 期望的tgo
            actual_tgo = []  # 实际的tgo
            impact_time_error = []  # tgo的误差
            state = env.get_state(td)
            episode_reward = 0
            done = False
            while done is False:
                # collect state, action and reward
                action = agent.get_action(state)  # get new action with old state
                done = env.step(action=float(action), h=0.1)
                state_ = env.get_state(td)  # get new state with new action
                reward = env.get_reward(td)  # get new reward with new action
                agent.store_transition(state, action, reward)  # train with old state
                state = state_  # update state
                episode_reward += reward

                desired_tgo.append(td - env.Y[0])
                actual_tgo.append(env.tgo)
                impact_time_error.append(td - env.Y[0] - env.tgo)

                # update ppo
                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(state_, done, GAMMA=GAMMA)
                    agent.update()
            # end of one episode
            # env.plot_data(figure_num=0)

            # use the terminal data to update once
            if len(agent.reward_buffer) != 0:
                agent.reward_buffer[-1] -= env.R + (td - env.Y[0]) ** 2
                agent.finish_path(state, done, GAMMA=GAMMA)
                agent.update()
            episode_reward -= env.R + (td - env.Y[0]) ** 2

            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Training | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[0], td - env.Y[0]))

            plt.figure()
            plt.subplots_adjust(hspace=0.6)
            plt.subplot(2, 1, 1)
            plt.plot(np.array(env.reY)[:, 0], np.array(desired_tgo)[:-1], 'k--', label='desired tgo')
            plt.plot(np.array(env.reY)[:, 0], np.array(actual_tgo)[:-1], 'k-', label='actual tgo')
            plt.xlabel('Time (s)')
            plt.ylabel('t_go(s)')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot()
            plt.plot(np.array(env.reY)[:, 0], np.array(impact_time_error)[:-1], 'k-')
            plt.xlabel('Time (s)')
            plt.ylabel('impact time error(s)')
            plt.grid()
            plt.show()

            # calculate the discounted episode reward
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * .9 + episode_reward * .1)

            # save the episode data
            dict_episode_reward['episode_reward'].append(episode_reward)
            dict_episode_reward['all_episode_reward'] = all_episode_reward
            dict_episode_reward['target_time'].append(td)
            dict_episode_reward['actual_time'].append(env.Y[0])

            # save model and data
            if episode_reward > REWARD_SAVE_CASE:
                REWARD_SAVE_CASE = episode_reward
                # if abs(td - env.Y[0]) < 0.5:
                agent.save_model('./ppo_model/agent{}'.format(model_num))
                savemat('./ppo_reward.mat', dict_episode_reward)
                model_num = (model_num + 1) % 20

        agent.save_model('./ppo_model/agent_end')
        savemat('./ppo_reward.mat', dict_episode_reward)

        plt.figure(1)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['PPO', time.strftime("%Y_%m%d_%H%M")])))
        plt.show()
    else:
        # test
        agent = PPO(state_dim, action_dim, action_bound, r'./ppo_model')
        flight_data = {}
        for episode in range(1):  # (TEST_EPISODES):
            # env.modify()
            env.modify([0., 200., 0, -20000., 20000, 200])
            td = 120  # env.get_tgo() * random.uniform(1.1, 1.2)
            desired_tgo = []  # 期望的tgo
            actual_tgo = []  # 实际的tgo
            impact_time_error = []  # tgo的误差
            state = env.get_state(td)
            action = 0
            episode_reward = 0
            done = False
            t = []
            while done is False:
                action = agent.get_action(state, greedy=True)  # use the mean of distribution as action

                # jeon2006impact
                # AC = 3 * env.v * env.qdot
                # K = -120 * env.v ** 5 / (AC * env.R ** 3)
                # tgo = (1 + (env.theta - env.q) ** 2 / 10) * env.R / env.v
                # # tgo = env.get_tgo()
                # action = K * ((td - env.t - tgo)

                # kim2013biased
                # AC = 3 * v * qdot
                # rgo = (1 + (theta - q) ** 2 / 10) * R
                # eps = v * (td - t) - rgo
                # action = AC * (0.5 - 0.5 * math.sqrt(1 + (240 * v ** 4 * eps / (AC ** 2 * R ** 3))))

                # tahk2018impact
                # lamb = env.theta - env.q  # 超前角
                # D = td - env.t  # 期望tgo
                # P = env.R / env.v * (1 + (lamb ** 2) / 10)  # 预测tgo
                # # P = env.get_tgo()
                # E = D - P  # 预测到达时间误差
                # action = -3 * env.v ** 2 / env.R * lamb + (5 * 20 * env.v ** 2) / (env.R * lamb) * (E / D)

                # action = np.clip(action, -action_bound, action_bound)

                done = env.step(action=action, h=0.1)
                state = env.get_state(td)
                reward = env.get_reward(td)
                episode_reward += reward

                desired_tgo.append(td - env.Y[0])
                actual_tgo.append(env.tgo)
                impact_time_error.append(td - env.Y[0] - env.tgo)

            env.plot_data(figure_num=0)
            flight_data['sim_{}'.format(episode)] = env.save_data()
            flight_data['time{}'.format(episode)] = {'desired_tgo': np.array(desired_tgo),
                                                     'actual_tgo': np.array(actual_tgo),
                                                     'impact_time_error': np.array(impact_time_error)}
            # print the result
            episode_reward = episode_reward / env.Y[0]  # calculate the average episode reward
            print('Testing | Episode: {}/{} | Average Episode Reward: {:.2f} | Running Time: {:.2f} | '
                  'Target Time: {:.2f} | Actual Time: {:.2f} | Error Time: {:.2f}'
                  .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0,
                          td, env.Y[0], td - env.Y[0]))
            # plt.figure()
            # plt.subplots_adjust(hspace=0.6)
            # plt.subplot(2, 1, 1)
            # plt.plot(np.array(env.reY)[:, 0], np.array(desired_tgo)[:-1], 'k--', label='desired tgo')
            # plt.plot(np.array(env.reY)[:, 0], np.array(actual_tgo)[:-1], 'k-', label='actual tgo')
            # plt.xlabel('Time (s)')
            # plt.ylabel('t_go(s)')
            # plt.legend()
            # plt.grid()
            #
            # plt.subplot(2, 1, 2)
            # plt.plot()
            # plt.plot(np.array(env.reY)[:, 0], np.array(impact_time_error)[:-1], 'k-')
            # plt.xlabel('Time (s)')
            # plt.ylabel('impact time error(s)')
            # plt.grid()
            # plt.show()
        savemat('./flight_sim_ppo.mat', flight_data)
