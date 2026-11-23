import numpy
import time
import wandb
import numpy as np
import torch
import datetime
from collections import deque

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
        self.use_n_step_search = self.args.use_n_step_search
        self.search_depth_N = self.args.search_depth_N

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

    @torch.no_grad()
    def _sample_search_actions(self, step):
        N = self.search_depth_N
        n_agents = self.num_agents
        n_threads = self.n_rollout_threads

        cent_obs = np.concatenate(self.buffer.share_obs[step])  # (n_thread*n_agent, obs_dim)
        obs = np.concatenate(self.buffer.obs[step])  # (n_thread*n_agent, obs_dim)
        masks = np.concatenate(self.buffer.masks[step])  # (n_thread*n_agent, 1)
        rnn_states = np.concatenate(self.buffer.rnn_states[step])  # unused
        rnn_states_critic = np.concatenate(self.buffer.rnn_states_critic[step])  # unused
        available_actions = np.concatenate(self.buffer.available_actions[step])

        policy = self.trainer.policy
        policy.eval()
        search_actions = []

        for _ in range(N):
            _, act, _, _, _ = policy.get_actions(
                cent_obs, obs, rnn_states, rnn_states_critic,
                masks, available_actions,
                deterministic=False  # 随机采样 [6]
            )
            search_actions.append(_t2n(act))  # np.ndarray (n_thread*n_agent,)

        policy.train()
        return search_actions

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []
        accumulated_reward = []

        feasible_rate_history = []
        rho_means = []
        pickle_infos = {"episode_reward": accumulated_reward,
                        "best_episode_reward": -float('inf'),
                        "best_episode_slots": None,
                        "best_episode_actions": None,
                        "convergence_episode": -1, "convergence_time": -1,
                        "feasible_rate_history": feasible_rate_history,
                        "rho_means": rho_means}

        episode = 0
        counter_for_real_done = 0
        feasible_window_size = 50
        feasible_window = deque(maxlen=feasible_window_size)

        # ========== 收敛判定参数 ==========
        # best_eval_reward = -float('inf')  # 历史最佳评估奖励
        feasible_rate_threshold = 0.98  # 可行率达标阈值
        improve_threshold = 0.02  # eval奖励相对提升小于2%视为无优化
        # eval_converge_patience = 2  # eval连续无提升轮数阈值
        patience_counter = 0  # eval耐心计数器

        # ========== 新增：训练侧停滞判定 ==========
        train_stall_threshold = 15  # 连续训练10回合无最佳奖励刷新即判定停滞
        train_stall_counter = 0  # 训练停滞计数器
        last_eval_stable = False  # 记录最近一次eval是否稳定
        is_converged = False

        best_episode_reward = -float('inf')
        best_episode_dwell_times = float('inf')

        current_feasible_rate = 0.0

        # start_time = time.time()

        while episode < episodes and not is_converged:
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            # 标记本训练回合是否刷新过最佳奖励
            episode_has_best_update = False
            # 本训练回合的 ep 级统计累加器
            round_ep_rewards = []
            round_ep_dws = []
            round_ep_rhos = []
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                search_actions = None
                if self.use_n_step_search:
                    search_actions = self._sample_search_actions(step)

                # 课程式随机概率：从 0 线性增长到 random_prob_max
                ep_start = self.all_args.random_prob_start
                ep_end = self.all_args.random_prob_end
                p_max = self.all_args.random_prob_max

                if episode < ep_start:
                    random_prob = 0.0
                else:
                    progress = min(1.0, (episode - ep_start) / (ep_end - ep_start))
                    random_prob = progress * p_max
                # random_prob = current_feasible_rate >= 0.90 and len(feasible_window) >= 20

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions,
                                                                                          random_prob=random_prob,
                                                                                          search_actions=search_actions)
                # self.envs.get_current_station()
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        # print(actions)
                        train_episode_rewards[t] = 0

                        # if infos[t]["real_done"] is True and self.teaching_force_mode[t] is False:
                        if infos[t]["real_done"] is True:
                            counter_for_real_done += 1
                            feasible_window.append(1)
                            ep_reward = np.sum(infos[t]["episode_reward"])
                            round_ep_rewards.append(ep_reward)
                            ep_dwell_times = np.sum(infos[t]['action'])
                            round_ep_dws.append(ep_dwell_times)
                            if "rho_history" in infos[t] and len(infos[t]["rho_history"]) > 0:
                                rho_mean_val = float(np.mean(infos[t]["rho_history"]))
                            else:
                                rho_mean_val = 0.0
                            round_ep_rhos.append(rho_mean_val)
                            # 只更新最佳奖励，标记本回合有更新
                            if ep_reward > best_episode_reward:
                                best_episode_reward = ep_reward
                                pickle_infos["best_episode_reward"] = best_episode_reward
                                episode_has_best_update = True
                            if ep_dwell_times < best_episode_dwell_times:
                                best_episode_dwell_times = ep_dwell_times
                                best_episode_slots = infos[t]["episode_time_slots"]
                                best_episode_actions = infos[t]["action"]
                                pickle_infos["best_episode_slots"] = best_episode_slots
                                pickle_infos["best_episode_actions"] = best_episode_actions
                        else:
                            feasible_window.append(0)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic
                # insert data into buffer
                self.insert(data)

            if len(feasible_window) >= 10:
                current_feasible_rate = sum(feasible_window) / len(feasible_window)
            else:
                current_feasible_rate = 0.0
            feasible_rate_history.append(current_feasible_rate)

            # ==================== 训练回合结束：统一判定最佳奖励停滞 ====================
            if episode_has_best_update:
                train_stall_counter = 0
            else:
                train_stall_counter += 1

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
                # save the best model
                if best_episode_reward > -float('inf'):
                    self.save(episode, suffix='_best')

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                if len(feasible_window) >= 10:
                    current_feasible_rate = sum(feasible_window) / len(feasible_window)
                else:
                    current_feasible_rate = 0.0

                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                print(
                    f"window feasible rate(window{len(feasible_window)}): {current_feasible_rate:.2%} |"
                    f" best reward in history: {best_episode_reward:.4f} | best dwell times: {best_episode_dwell_times}\n")
                train_infos["current_feasible_rate"] = current_feasible_rate
                train_infos["best_episode_reward"] = best_episode_reward
                round_mean_reward = float(np.mean(round_ep_rewards))
                round_mean_rho = float(np.mean(round_ep_rhos))
                train_infos["episode_mean_reward"] = round_mean_reward
                train_infos["episode_mean_rho"] = round_mean_rho
                train_infos["best_episode_dwell_times"] = best_episode_dwell_times
                self.log_train(train_infos, total_num_steps)

                # if len(done_episodes_rewards) > 0:
                #     aver_episode_rewards = np.mean(done_episodes_rewards)
                #     self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards},
                #                              total_num_steps)
                #     done_episodes_rewards = []

            # ==================== eval阶段收敛判断 ====================
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     eval_avg_reward, eval_feasible_rate, eval_stable = self.eval(total_num_steps)
            #     last_eval_stable = eval_stable  # 记录本次eval的稳定性
            #
            #     # eval侧同步更新最佳奖励，辅助判断
            #     if eval_feasible_rate >= feasible_rate_threshold:
            #         if best_eval_reward == -float('inf'):
            #             best_eval_reward = eval_avg_reward
            #             patience_counter = 0
            #             self.save(episode, suffix='_eval_best')
            #         else:
            #             improve_rate = (eval_avg_reward - best_eval_reward) / abs(best_eval_reward)
            #             if improve_rate >= improve_threshold:
            #                 best_eval_reward = eval_avg_reward
            #                 patience_counter = 0
            #                 self.save(episode, suffix='_eval_best')
            #             else:
            #                 patience_counter += 1

            # ==================== 实时收敛判定（每回合都检查） ====================
            if not is_converged:
                # 条件1：可行率达标
                feasible_ok = len(feasible_window) >= 15 and current_feasible_rate >= feasible_rate_threshold
                # 条件2：训练最佳奖励连续停滞（回合级）
                train_stall_ok = train_stall_counter >= train_stall_threshold
                # # 条件3：最近一次eval稳定
                # eval_stable_ok = last_eval_stable

                if feasible_ok and train_stall_ok:
                    is_converged = True
                    pickle_infos["convergence_episode"] = episode
                    pickle_infos["convergence_time"] = time.time() - start
                    print(f"\nThe training is completed：no best cumulative reward has been updated in consecutive {train_stall_counter} episodes")
                    print(
                        f"convergence episode：{episode} | Best cumulative reward in history：{best_episode_reward:.4f} |"
                        f" window feasible rate：{current_feasible_rate:.2%}")

            episode += 1

        # save results including episode_time_slots, episode_reward and action
        # file_name = self.run_dir + "statistics.pkl"
        pickle_infos["feasible_rate_history"] = feasible_rate_history
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
        eval_feasible_count = 0  # 统计eval中的可行次数
        eval_rho_list = []

        # ========== 新增：收敛检测参数 ==========
        stable_window_size = 5  # 连续检测窗口大小
        fluctuation_threshold = 0.15  # 最大允许相对波动（15%）
        eval_stable = False  # 本轮eval是否满足稳定收敛

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
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava = self.eval_envs.step(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards
            # 单步得分
            eval_scores = eval_rewards
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
                    # 统计可行解
                    if eval_infos[eval_i]["real_done"]:
                        eval_feasible_count += 1
                        # 新增：收集 rho
                        if "rho_history" in eval_infos[eval_i]:
                            rho_h = eval_infos[eval_i]["rho_history"]
                            if len(rho_h) > 0:
                                eval_rho_list.append(float(np.mean(rho_h)))
                    # ========== 滑动窗口检测奖励稳定性 ==========
                    if len(eval_episode_rewards) >= stable_window_size and not eval_stable:
                        window = eval_episode_rewards[-stable_window_size:]
                        mean_win = np.mean(window)
                        # 避免均值接近0导致除零异常
                        if abs(mean_win) < 1e-6:
                            max_min_diff = np.max(window) - np.min(window)
                            stable = max_min_diff < 1e-3
                        else:
                            fluctuation = (np.max(window) - np.min(window)) / abs(mean_win)
                            stable = fluctuation <= fluctuation_threshold
                        if stable:
                            eval_stable = True

            if eval_episode >= self.all_args.eval_episodes:
                key_average = 'eval/average_episode_rewards'
                key_max = 'eval/max_episode_rewards'
                key_scores = 'eval/average_episode_scores'
                key_feasible = 'eval/feasible_rate'
                key_rho = 'eval/mean_rho'

                eval_feasible_rate = eval_feasible_count / self.all_args.eval_episodes
                eval_mean_rho = float(np.mean(eval_rho_list)) if eval_rho_list else 0.0

                eval_env_infos = {
                    key_average: eval_episode_rewards,
                    key_max: [np.max(eval_episode_rewards)],
                    key_scores: eval_episode_scores,
                    key_feasible: [eval_feasible_rate],
                    key_rho: [eval_mean_rho]
                }
                self.log_env(eval_env_infos, total_num_steps)
                print("eval average episode rewards: {}, eval feasible rate: {:.2%}."
                      .format(np.mean(eval_episode_rewards), eval_feasible_rate))
                return np.mean(eval_episode_rewards), eval_feasible_rate, eval_stable





