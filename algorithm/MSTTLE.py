import copy
from gymnasium.spaces import Discrete, Box
from env_util import *


class TTLE:
    def __init__(self, _cfg):

        self.rho_history = []
        self.cfg = _cfg
        self.n_agents = self.cfg.numT
        self.optional_actions = self.cfg.ava_actions
        self._action_shape = len(self.optional_actions)

        self.search_depth = self.cfg.search_depth_N
        self.use_n_step_search = self.cfg.use_n_step_search
        self.use_expert_rule = self.cfg.use_expert_rule

        self.stop_plan = np.array(self.cfg.stop_plan)
        assert len(self.stop_plan) == self.n_agents

        self.shuffled_stop_plan = None

        self.use_action_mask = False

        self.state, _, _ = self.reset()

        self._obs_shape = (len(self.state[0]),)

        self.penalty = self.optional_actions[-1] * 10

        self.observation_space = [Box(low=float("-inf"), high=float("inf"), shape=self._obs_shape,
                                      dtype=np.float32) for _ in range(self.n_agents)]
        self.share_observation_space = self.observation_space
        self.action_space = [Discrete(self._action_shape) for _ in range(self.n_agents)]

        self.current_station = 1

        self.weight_dwell = _cfg.weight_dwell
        self.weight_adjust = _cfg.weight_adjust
        self.weight_multistep = _cfg.weight_multistep

        self.checkCons = np.array([0 for _ in range(self.n_agents)])

        self.test_mode = False

        self.stop_info = []
        self.time_slots = []
        self.time_slots.append(self.cfg.startTime)
        self.chosen_action = []
        self.accumulated_reward = np.zeros(self.n_agents)

        self.enable_sanity_check = False

    def reset(self, seed=None, options=None):

        self.current_station = 1

        self.stop_info = []
        self.time_slots = []
        self.time_slots.append(self.cfg.startTime)
        self.chosen_action = []
        self.accumulated_reward = np.zeros(self.n_agents)
        self.rho_history = []

        ifStop = [1 for _ in range(self.cfg.numT)]

        state = []
        for i in range(self.cfg.numT):
            time_s = self.cfg.startTime[i] / self.cfg.timeZone
            direction_s = self.cfg.direction[i]
            stop_s = ifStop[i]
            state.append(np.array((time_s, direction_s, stop_s)))
        state = np.array(state, dtype=np.float32)

        if self.use_action_mask is True:
            available_action = get_available_actions(self.n_agents, self.optional_actions, self.shuffled_stop_plan.T[0])
        else:
            available_action = np.ones((self.n_agents, len(self.optional_actions)))

        return state, state, available_action

    def _simulate_one_step(self, state, action, station_idx):
        tmp_state = self._obs_wrapper_for_mat(state)
        ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr = \
            get_time(tmp_state, action, station_idx, self.cfg)

        check = check_all_cons((ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr),
                               self.cfg)

        return check, ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr

    def _build_next_state(self, next_dep, ifStopNext):
        state = []
        for i in range(self.n_agents):
            time_s = next_dep[i] / self.cfg.timeZone
            direction_s = self.cfg.direction[i]
            stop_s = ifStopNext[i]
            state.append(np.array((time_s, direction_s, stop_s), dtype=np.float32))
        return state

    def _n_step_tree_search(self, if_stop_pre, pre_dep, station_idx, action_sequence=None):
        """
        前向搜索 N 步冲突检测。
        """
        if action_sequence is not None:
            N = len(action_sequence)
        else:
            N = self.search_depth

        max_valid = len(self.cfg.upRunTime)  # 有效区间总数
        remaining = max_valid - station_idx
        N = min(N, remaining)
        if N <= 0:
            return np.zeros(self.n_agents)

        # 同步截断 action_sequence
        if action_sequence is not None:
            action_sequence = action_sequence[:N]

        current_state = self._build_next_state(pre_dep, if_stop_pre)
        conflict_count = np.zeros(self.n_agents)

        for t in range(N):
            if action_sequence is not None:
                # action_sequence[t] 形状: (n_thread*n_agent,) → 取当前 env 的 slice
                # 单线程时 n_thread=1，直接 reshape 为 (n_agent,)
                raw = action_sequence[t]
                if raw.size > self.n_agents:
                    action = raw[:self.n_agents]
                else:
                    action = raw
                action = self._action_wrapper(action)
            else:
                sampled_indices = np.random.randint(0, len(self.optional_actions), size=self.n_agents)
                action = np.array([self.optional_actions[idx] for idx in sampled_indices])

            check, ifStopPre, ifStopNext, pre_dep_new, pre_arr, next_dep, next_arr = \
                self._simulate_one_step(current_state, action, station_idx + t)

            conflict_count += (check > 0).astype(np.float32)
            current_state = self._build_next_state(next_dep, ifStopNext)

        return conflict_count / N

    def _adjust_action_for_next_block(self, raw_action, current_station,
                                      random_prob, test_mode):
        adjusted_action = raw_action.copy().astype(np.int32)
        # print(adjusted_action)
        # adjust_amount = np.zeros(self.n_agents, dtype=np.int32)
        max_iter = 20
        # success = True
        cfg = self.cfg
        block_idx = current_station - 1
        next_block_idx = current_station
        # if next_block_idx >= self.cfg.numB - 1:
        #     return adjusted_action, adjust_amount, success
        state_flat = self._obs_wrapper_for_mat(self.state)

        if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = \
            get_time(state_flat, adjusted_action, next_block_idx, cfg, mode='future')

        times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
        _times, _actions, success, infos = repair_future_conflicts(times, adjusted_action, block_idx, cfg,
                                                                   random_prob=random_prob,
                                                                   test_mode=test_mode)
        return _times, _actions, success, np.array(infos)

    def _sanity_check_times(self, times_dict, stage_name):
        """
        硬检查时间数组合法性：无NaN/Inf、非负、不超出时域范围
        :param times_dict: {变量名: 时间数组}
        :param stage_name: 当前阶段名称，用于报错定位
        """
        if not self.enable_sanity_check:
            return
        for name, arr in times_dict.items():
            if arr is None:
                continue
            # 1. NaN/Inf 检查
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                nan_idx = np.where(np.isnan(arr) | np.isinf(arr))[0]
                raise AssertionError(
                    f"[阶段: {stage_name}] 时间变量「{name}」包含 NaN/Inf！\n"
                    f"当前车站: {self.current_station}\n"
                    f"异常列车索引: {nan_idx}\n"
                    f"对应数值: {arr[nan_idx]}\n"
                    f"对应方向: {np.array(self.cfg.direction)[nan_idx]}"
                )
            # 2. 负时间检查
            if np.any(arr < 0):
                neg_idx = np.where(arr < 0)[0]
                raise AssertionError(
                    f"[阶段: {stage_name}] 时间变量「{name}」出现负时间！\n"
                    f"当前车站: {self.current_station}\n"
                    f"异常列车索引: {neg_idx}\n"
                    f"对应时间值: {arr[neg_idx]}\n"
                    f"对应方向: {np.array(self.cfg.direction)[neg_idx]}"
                )
            # 3. 时域越界检查
            if np.any(arr > self.cfg.timeZone):
                over_idx = np.where(arr > self.cfg.timeZone)[0]
                raise AssertionError(
                    f"[阶段: {stage_name}] 时间变量「{name}」超出时域上限！\n"
                    f"当前车站: {self.current_station}\n"
                    f"时域上限: {self.cfg.timeZone}\n"
                    f"异常列车索引: {over_idx}\n"
                    f"对应时间值: {arr[over_idx]}"
                )

    def _sanity_check_action(self, actions, stage_name):
        """
        硬检查动作合法性：形状正确、无NaN/Inf、在可选动作范围内
        """
        if not self.enable_sanity_check:
            return
        # 1. 形状校验
        if actions.shape != (self.n_agents,):
            raise AssertionError(
                f"[阶段: {stage_name}] 动作形状不匹配！\n"
                f"预期形状: ({self.n_agents},), 实际形状: {actions.shape}"
            )
        # 2. NaN/Inf 校验
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            raise AssertionError(
                f"[阶段: {stage_name}] 动作包含 NaN/Inf！\n"
                f"当前车站: {self.current_station}\n"
                f"动作值: {actions}"
            )
        # 3. 可选范围校验
        valid_mask = np.isin(actions, self.optional_actions)
        if not np.all(valid_mask):
            bad_idx = np.where(~valid_mask)[0]
            raise AssertionError(
                f"[阶段: {stage_name}] 动作超出可选范围！\n"
                f"当前车站: {self.current_station}\n"
                f"可选动作集: {self.optional_actions}\n"
                f"异常列车索引: {bad_idx}\n"
                f"非法动作值: {actions[bad_idx]}"
            )

    def step(self, actions, random_prob=0.0, test_mode=False, search_actions=None):

        # raw_actions = actions
        raw_actions = self._action_wrapper(actions)

        weight_dwell = self.weight_dwell
        weight_adjust = self.weight_adjust
        weight_multistep = self.weight_multistep

        reward = []
        info = {}

        if self.use_expert_rule:
            adjust_times, adjust_actions, adjust_success, conflict_infos = self._adjust_action_for_next_block(
                raw_actions, self.current_station, random_prob=random_prob, test_mode=test_mode)
            if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = adjust_times

            # adj_times = {
            #     "pre_dep_adj": pre_dep,
            #     "pre_arr_adj": pre_arr,
            #     "next_dep_adj": next_dep,
            #     "next_arr_adj": next_arr
            # }
            # self._sanity_check_times(adj_times, "专家规则调整后")
            # self._sanity_check_action(adjust_actions, "专家规则调整后")

            if adjust_success == 'failure':
                # print(f'failure infos: {conflict_infos}')
                self.time_slots.append(pre_arr)
                self.time_slots.append(next_dep)
                self.time_slots.append(next_arr)
                info = {"real_done": False}
                info["episode_time_slots"] = np.array(self.time_slots)
                state, _, available_action = self.reset()
                reward = [-1.0 if i in conflict_infos else adjust_actions[i] / self.optional_actions[-1]
                          for i in range(self.n_agents)]
                self.accumulated_reward = np.zeros(self.n_agents)
                shared_obs = copy.deepcopy(state)
                reward = np.array(reward, dtype=np.float32).reshape(self.n_agents, 1)
                done = np.array([True for _ in range(self.n_agents)])
                self.state = state
                return state, shared_obs, reward, done, info, available_action
        else:
            _conflict_infos, if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = \
                self._simulate_one_step(self.state, raw_actions, self.current_station)
            adjust_actions = raw_actions
            conflict_infos = np.where(_conflict_infos > 0)[0]
            if len(conflict_infos) > 0:
                info = {"real_done": False}
                self.time_slots.append(pre_arr)
                self.time_slots.append(next_dep)
                info["episode_time_slots"] = np.array(self.time_slots)
                state, _, available_action = self.reset()
                reward = [-1.0 if i in conflict_infos else raw_actions[i] / self.optional_actions[-1]
                          for i in range(self.n_agents)]
                self.accumulated_reward = np.zeros(self.n_agents)
                shared_obs = copy.deepcopy(state)
                reward = np.array(reward, dtype=np.float32).reshape(self.n_agents, 1)
                done = np.array([True for _ in range(self.n_agents)])
                self.state = state
                return state, shared_obs, reward, done, info, available_action

        rho = np.zeros((self.n_agents,))
        if self.use_n_step_search and self.current_station < self.cfg.numS - 2:
            rho = self._n_step_tree_search(
                if_stop_pre, next_dep, self.current_station + 1,
                action_sequence=search_actions)
        self.rho_history.append(float(np.mean(rho)))

        for i in range(self.n_agents):
            if i in conflict_infos:
                _r = -1.0
            else:
                punish_dwell = raw_actions[i] / self.cfg.ava_actions[-1]
                punish_adjust = abs(adjust_actions[i] - raw_actions[i]) / self.cfg.ava_actions[-1]
                punish_multistep = rho[i]
                total_penalty = punish_dwell * weight_dwell + punish_adjust * weight_adjust +\
                                punish_multistep * weight_multistep
                _r = -total_penalty
            _r = max(-1.0, min(_r, 1.0))
            reward.append(_r)
            self.accumulated_reward[i] += _r

        self.time_slots.append(pre_arr)
        self.time_slots.append(next_dep)
        self.stop_info.append(if_stop_pre)
        self.stop_info.append(if_stop_next)
        self.chosen_action.append(adjust_actions)

        self.current_station += 1

        state = []
        info["real_done"] = False
        for i in range(self.n_agents):
            time_s = next_dep[i] / self.cfg.timeZone
            direction_s = self.cfg.direction[i]
            stop_s = if_stop_next[i]
            state.append(np.array((time_s, direction_s, stop_s), dtype=np.float32))
        state = np.array(state, dtype=np.float32)
        done = np.array([False for _ in range(self.n_agents)])

        if self.use_action_mask and self.current_station - 1 < len(self.stop_plan.T):
            available_action = get_available_actions(
                self.n_agents, self.optional_actions,
                self.stop_plan.T[self.current_station - 1]
            )
        else:
            available_action = np.ones((self.n_agents, len(self.optional_actions)))

        if self.current_station == self.cfg.numS - 1:
            done = np.array([True for _ in range(self.n_agents)])
            info["real_done"] = True
            info["episode_reward"] = self.accumulated_reward
            self.time_slots.append(next_arr)
            info["episode_time_slots"] = np.array(self.time_slots)
            info["action"] = self.chosen_action
            info["rho_history"] = self.rho_history
            state, _, available_action = self.reset()

        self.state = state
        shared_obs = copy.deepcopy(state)
        reward = np.array(reward, dtype=np.float32).reshape(self.n_agents, 1)

        return state, shared_obs, reward, done, info, available_action

    def _action_wrapper(self, action):
        action = np.array([self.optional_actions[int(a)] for a in action])
        return action

    def _obs_wrapper_for_mat(self, state):
        return copy.deepcopy(np.array(state, dtype=np.float32).flatten())

    def get_env_info(self):
        env_info = {"state_shape": self.observation_space[0].shape,
                    "obs_shape": self.observation_space[0].shape,
                    "n_actions": self.action_space[0].n,
                    "n_agents": self.n_agents,
                    "action_spaces": self.action_space
                    }
        return env_info

    def close(self):
        pass


