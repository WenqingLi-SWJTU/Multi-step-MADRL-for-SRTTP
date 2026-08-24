import numpy
import time
import wandb
import numpy as np
import torch
import datetime

from base_runner import Runner
import pickle


def _t2n(x):
    return x.detach().cpu().numpy()


class TTLE_Runner(Runner):
    def __init__(self, config):
        super(TTLE_Runner, self).__init__(config)

        self.args = config['all_args']
        self.exp_name = self.args.scenario
        self.run_dir = config['run_dir']

        # if self.use_teaching_force:
        #     self.teaching_force_counter = np.zeros(self.n_rollout_threads)
        #     self.teaching_force_mode = np.zeros(self.n_rollout_threads, dtype=np.bool)
        #     self.teaching_action = np.zeros_like(np.array(self.all_args.teaching_action))
        #     for i in range(self.teaching_action.shape[0]):
        #         for j in range(self.teaching_action.shape[1]):
        #             self.teaching_action[i, j] = np.where(
        #                 self.all_args.ava_actions == np.array(self.all_args.teaching_action[i][j]))[0].squeeze()
        # keys = list(range(self.n_rollout_threads))
        # values = [None for _ in range(self.n_rollout_threads)]
        # self.teaching_info = {}
        # for key, value in zip(keys, values):
        #     self.teaching_info[key] = value

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        time_slots = []
        action = []
        accumulated_reward = []
        pickle_infos = {"episode_time_slots": time_slots, "action": action,
                        "episode_reward": accumulated_reward,
                        "convergence_episode": -1, "convergence_time": -1}

        episode = 0
        counter_for_real_done = 0

        # start_time = time.time()

        stop_iteration_threshold = 50

        while episode < episodes and counter_for_real_done < stop_iteration_threshold:
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                # if self.use_teaching_force:
                #     for i in range(self.n_rollout_threads):
                #         if self.envs.get_current_station()[i] - 1 == 0:
                #             self.teaching_force_mode[i] = True if np.random.random() < self.teaching_force_rate else\
                #                 False
                #
                #         if self.teaching_force_mode[i]:
                #             self.teaching_info[i] = self.teaching_action[self.envs.get_current_station()[i] - 1]
                #         else:
                #             self.teaching_info[i] = None

                # print(f"teaching_force_mode: {self.teaching_force_mode}")

                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # self.envs.get_current_station()

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        # print(actions)
                        train_episode_rewards[t] = 0
                        if infos[t]["real_done"] is True:
                        # if infos[t]["real_done"] is True and self.teaching_force_mode[t] is False:
                            counter_for_real_done += 1
                            print("Thread {}: episode {} has real done\n time_slot: {}\n reward: {}"
                                  .format(t, episode, infos[t]["episode_time_slots"], np.sum(infos[t]["episode_reward"])))
                            pickle_infos["episode_time_slots"].append(infos[t]["episode_time_slots"])
                            pickle_infos["action"].append(infos[t]["action"])
                            pickle_infos["episode_reward"].append(infos[t]["episode_reward"])
                            _time_slots = infos[t]["episode_time_slots"]
                            if counter_for_real_done >= stop_iteration_threshold:
                                self.save(episode)
                                break

                if counter_for_real_done >= stop_iteration_threshold:
                    self.save(episode)
                    break

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            attn_score = train_infos['attn_score']
            # file_name = 'attn_score_' + self.exp_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.pkl'
            file_name = 'attn_score_' + self.exp_name + '.pkl'
            # with open(file_name, 'wb') as f:
            #     pickle.dump(attn_score, f)
            del train_infos['attn_score']

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # print(f"time slots: \n{_time_slots}")

                self.log_train(train_infos, total_num_steps)

                # if len(done_episodes_rewards) > 0:
                #     aver_episode_rewards = np.mean(done_episodes_rewards)
                #     self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                #                              total_num_steps)
                #     done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            episode += 1

        # save results including episode_time_slots, episode_reward and action
        # file_name = self.run_dir + "statistics.pkl"
        file_name = "statistics.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(pickle_infos, f)

    def warmup(self):
        # reset env
        obs, share_obs, ava = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = ava.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # if self.all_args.use_teaching_force is True:
        #     self.teaching_action = np.array(self.all_args.teaching_action)
        #     teaching_action = np.zeros(self.teaching_action.shape)
        #     ava_actions = self.all_args.ava_actions
        #     for i in range(teaching_action.shape[0]):
        #         for j in range(teaching_action.shape[1]):
        #             teaching_action[i, j] = np.where(ava_actions == self.teaching_action[i][j])[0].squeeze()
        #     rand_num = np.random.random()
        #     if rand_num < self.all_args.teaching_force_rate:
        #         self.teaching_force_mode = True
        #         n_section = self.envs.get_current_station() - 1
        #         self.teaching_info = {"teaching_action": teaching_action,
        #                               "n_section": n_section}
        #     else:
        #         self.teaching_force_mode = False
        #         self.teaching_info = None
        # else:
        #     self.teaching_force_mode = False
        #     self.teaching_info = None

        value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),   # shape: (n_thread * n_agent, obs_dim)
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),   # unused
            np.concatenate(self.buffer.rnn_states_critic[step]),   # unused
            np.concatenate(self.buffer.masks[step]),
            # self.teaching_info,
            np.concatenate(self.buffer.available_actions[step])
            )

        # [n_thread, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        # print(actions)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, ava = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.all_args.eval_episodes, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(ava),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.eval_episodes))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step_20(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_scores = [t_info[0]["score_reward"] for t_info in eval_infos]
            one_episode_scores += np.array(eval_scores)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

                    eval_episode_scores.append(one_episode_scores[eval_i])
                    one_episode_scores[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = '/eval_average_episode_rewards'
                key_max = '/eval_max_episode_rewards'
                key_scores = '/eval_average_episode_scores'
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)],
                                  key_scores: eval_episode_scores}
                self.log_env(eval_env_infos, total_num_steps)

                print("eval average episode rewards: {}, scores: {}."
                      .format(np.mean(eval_episode_rewards), np.mean(eval_episode_scores)))
                break
