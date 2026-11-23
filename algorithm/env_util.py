import numpy as np


def get_single_time(_id, pre_dep, if_stop_pre, action, if_stop_next, n_section, config):
    direction = config.direction[_id]
    up_runtime = config.upRunTime[n_section]
    down_runtime = config.downRunTime[n_section]
    time_loss_of_ac = config.timeLossOfAc
    time_loss_of_dc = config.timeLossOfDc

    if direction == 0:  # 下行列车
        pre_arr = pre_dep + if_stop_pre * time_loss_of_ac + if_stop_next * time_loss_of_dc + \
                  down_runtime
        if action is not None:
            next_dep = pre_arr + action
        else:
            next_dep = None
    else:
        pre_arr = pre_dep - if_stop_pre * time_loss_of_dc - if_stop_next * time_loss_of_ac - \
                  up_runtime
        if action is not None:
            next_dep = pre_arr - action
        else:
            next_dep = None

    return pre_arr, next_dep


def obs_to_time(states, config):
    timeZone = config.timeZone
    numT = config.numT
    pre_dep = [np.round(states[3 * i] * timeZone) for i in range(numT)]  # 在上一车站（当前区间的后方站）的到达时刻
    direction = [states[3 * i + 1] for i in range(numT)]  # 列车运行方向
    ifStopPre = [states[3 * i + 2] for i in range(numT)]  # 在上一车站（当前区间的后方站）是否停车
    return pre_dep, direction, ifStopPre


# def get_time(states, actions, current_station, timeZone, numT, numB, downRunTime, upRunTime):
def get_time(states, actions, current_station, config,
             mode='current'):
    numT = config.numT
    numB = config.numB
    pre_dep, direction, ifStopPre = obs_to_time(states, config)
    pre_dep = np.array(pre_dep, dtype=np.int16)
    ifStopPre = np.array(ifStopPre, dtype=np.int8)
    pre_arr = np.array([0 for _ in range(numT)], dtype=np.int16)  # 在当前车站（当前区间的前方站）的到达时刻
    next_dep = np.array([0 for _ in range(numT)], dtype=np.int16)  # 在当前车站（下一区间的后方站）的出发时刻
    ifStopNext = np.array([1 if actions[i] > 0 else 0 for i in range(numT)], dtype=np.int8)  # 在当前车站（下一区间的后方站）是否停车
    next_arr = None
    location = current_station  # 当前车站编号
    n_section = int(location) - 1  # 当前区间编号
    for i in range(numT):
        pre_arr[i], next_dep[i] = get_single_time(i, pre_dep[i], ifStopPre[i], actions[i],
                                                  ifStopNext[i], n_section, config)
    if n_section >= numB - 2 or mode == "future":
        next_arr = [0 for _ in range(numT)]  # 在下一车站（下一区间的前方站）即终到站的到达时刻
        for i in range(numT):
            next_arr[i], _ = get_single_time(i, next_dep[i], ifStopNext[i], None, 1,
                                             n_section + 1, config=config)
        next_arr = np.array(next_arr, dtype=np.int16)
    return ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr


# ================================================================
# check cross constraints between two trains in the section
# return a bool variable denotes whether these trains disobey the cross constraint
# ================================================================
def check_cross_cons(times):  # 判断两线段是否相交
    # formD: 前发车在当前区间后方站的出发时刻
    # formA: 前发车在当前区间前方站的到达时刻
    # laterD: 后发车在当前区间后方站的出发时刻
    # laterA: 后发车在当前区间前方站的到达时刻
    formD, formA, laterD, laterA = times[0], times[1], times[2], times[3]
    deltaD = formD - laterD
    deltaA = formA - laterA
    D = (deltaD > 0 and deltaA < 0) or (deltaD < 0 and deltaA > 0)
    return D


# ================================================================
# check all trains according to station and section constraints
# return the number of trains who disobey any constraint (i.e. sHeadway, cHeadway or crossover constraint)
# ================================================================
def check_sta_cons(time):
    formD, formA, laterD, laterA, next_formD, next_laterD = \
        time[0], time[1], time[2], time[3], time[4], time[5]
    staHeadway = 4
    check = 0
    formDirection = 0 if formD < formA else 1
    laterDirection = 0 if laterD < laterA else 1
    if formDirection == 0 and laterDirection == 1:
        if abs(next_laterD - formA) < staHeadway:
            check =  1
    return check


def check_sec_cons(time):
    formD, formA, laterD, laterA, next_formD, next_laterD = \
        time[0], time[1], time[2], time[3], time[4], time[5]
    check = 0
    secHeadway = 2
    formDirection = 0 if formD < formA else 1
    laterDirection = 0 if laterD < laterA else 1

    if formDirection == 0 and laterDirection == 1:
        if formA < next_laterD and next_laterD is not None:
            if abs(next_laterD - next_formD) < secHeadway:
                check = 1
        else:
            if abs(formA - laterA) < secHeadway:
                check = 1
    return check


def check_stat_and_sect_cons(time):
    staHeadway = 4
    secHeadway = 2
    checkS = 0  # 违背车站约束的列车数量
    formD, formA, laterD, laterA, next_formD, next_laterD = \
        time[0], time[1], time[2], time[3], time[4], time[5]
    formDirection = 0 if formD < formA else 1
    laterDirection = 0 if laterD < laterA else 1
    if formDirection == 0 and laterDirection == 1:
        if next_laterD is None and next_formD is None:
            if laterA - formA < secHeadway:
                checkS += 1
        else:
            if abs(next_laterD - formA) < staHeadway:
                checkS += 1

            if next_formD != formA:
                if abs(next_laterD - next_formD) < secHeadway:
                    checkS += 1
            elif next_laterD != laterA:
                if abs(formA - laterA) < secHeadway:
                    checkS += 1
    return checkS


# ================================================================
# check all trains according to consecutive constraints
# return the number of trains who disobey consecutive constraint
# ================================================================
def check_consec_cons(time, stopLater):
    checkC = 0
    stopDownLater = stopLater[0]
    stopUpLater = stopLater[1]
    cHeadwayWhenLaterStop = 2
    cHeadwayWhenLaterPass = 4
    formD, formA, laterD, laterA = time[0], time[1], time[2], time[3]
    formDirection = 0 if formD < formA else 1
    laterDirection = 0 if laterD < laterA else 1
    if formDirection == 0 and laterDirection == 0:
        if stopDownLater == 1:  # 后车停站
            if laterD - formA < cHeadwayWhenLaterStop:
                checkC += 1
        else:  # 后车通过
            if laterD - formA < cHeadwayWhenLaterPass:
                checkC += 1
    elif formDirection == 1 and laterDirection == 1:
        if stopUpLater == 1:  # 后车停站
            if laterA - formD < cHeadwayWhenLaterStop:
                checkC += 1
        else:  # 后车通过
            if laterA - formD < cHeadwayWhenLaterPass:
                checkC += 1
    return checkC


# ================================================================
# check two neighbouring trains according to all constraints (cross, consecutive constraints)
# with respect to potential departure and arrival time in the next section.
# return an array about which train disobey the constraints.
# ================================================================
def check_inblock_cons(times, config):
    check = []
    if_stop_pre, if_stop_next, pre_arr, next_dep, next_arr = times
    secHeadway = config.secHeadway
    cHeadwayWhenLaterStop = config.cHeadwayWhenLaterStop
    cHeadwayWhenLaterPass = config.cHeadwayWhenLaterPass
    # get the order of both trains
    if next_dep[0] != next_dep[1]:
        former_train = 0 if next_dep[0] < next_dep[1] else 1
        later_train = 0 if next_dep[0] > next_dep[1] else 1
    elif next_arr[0] != next_arr[1]:
        former_train = 0 if next_arr[0] < next_arr[1] else 1
        later_train = 0 if next_arr[0] > next_arr[1] else 1
    else:
        former_train = 0 if pre_arr[0] < pre_arr[1] else 1
        later_train = 0 if pre_arr[0] > pre_arr[1] else 1
    former_dir = 0 if next_dep[former_train] < next_arr[former_train] else 1
    later_dir = 0 if next_dep[later_train] < next_arr[later_train] else 1
    # crossover and overtaking headway
    _times = (next_dep[former_train], next_arr[former_train], next_dep[later_train], next_arr[later_train])
    if check_cross_cons(_times):
        if former_dir != later_dir:
            check.append('cross')
        else:
            check.append('overtaking')
    if former_dir == 0 and later_dir == 0:
        # both trains are downstream, check consecutive constraints
        if if_stop_pre[later_train]:
            if next_dep[later_train] - next_arr[former_train] < cHeadwayWhenLaterStop:
                check.append('consec')
        else:
            if next_dep[later_train] - next_arr[former_train] < cHeadwayWhenLaterPass:
                check.append('consec')
    if former_dir == 1 and later_dir == 1:
        if if_stop_next[later_train]:
            if next_arr[later_train] - next_dep[former_train] < cHeadwayWhenLaterStop:
                check.append('consec')
        else:
            if next_arr[later_train] - next_dep[former_train] < cHeadwayWhenLaterPass:
                check.append('consec')
    # one scenario of section headway
    if former_dir == 0 and later_dir == 1:
        # add buffer time
        # buffer_time = config.timeLossOfDc
        if abs(next_arr[former_train] - next_arr[later_train]) < secHeadway:
            check.append('fd_sec')
    return check


def check_backstation_cons(times, config):
    # when action is known, check staHeadway and secHeadway with known next_dep.
    check = []
    pre_arr, next_dep, next_arr = times
    secHeadway = config.secHeadway
    staHeadway = config.staHeadway
    # get the order of both trains
    if next_dep[0] != next_dep[1]:
        former_train = 0 if next_dep[0] < next_dep[1] else 1
        later_train = 0 if next_dep[0] > next_dep[1] else 1
    elif next_arr[0] != next_arr[1]:
        former_train = 0 if next_arr[0] < next_arr[1] else 1
        later_train = 0 if next_arr[0] > next_arr[1] else 1
    else:
        former_train = 0 if pre_arr[0] < pre_arr[1] else 1
        later_train = 0 if pre_arr[0] > pre_arr[1] else 1
    # get directions of both trains
    former_dir = 0 if next_dep[former_train] < next_arr[former_train] else 1
    later_dir = 0 if next_dep[later_train] < next_arr[later_train] else 1
    if former_dir != later_dir:
        if former_dir == 1 and later_dir == 0:
            if abs(next_dep[former_train] - pre_arr[later_train]) < staHeadway:
                check.append('sta')
        elif former_dir == 0 and later_dir == 1:
            if abs(next_dep[later_dir] - pre_arr[former_dir]) < staHeadway:
                check.append('sta')
    if former_dir == 1 and later_dir == 0:
        if next_dep[later_train] - next_dep[former_train] < secHeadway:
            check.append('bkd_sec')
    return check


# ================================================================
# check two trains according to future constraints (cross, consecutive, station and section constraints)
# with respect to arrival time in the current block and departure/potential arrival time in the next block.
# return an array about constraints types.
# ================================================================
def _check_future_cons(time_pair, config):
    if_stop_next, pre_arr, next_dep, next_arr = time_pair
    # get the order of both trains
    time_pair_for_bksta_check = ([pre_arr[0], pre_arr[1]],
                                  [next_dep[0], next_dep[1]],
                                  [next_arr[0], next_arr[1]])
    bksta_cons_check = check_backstation_cons(time_pair_for_bksta_check,
                                              config)
    # check in-block constraints in the next block assuming trains stop at the forward-station
    time_pair_for_inblock_check = ([if_stop_next[0], if_stop_next[1]],
                                    [1, 1],
                                    [pre_arr[0], pre_arr[1]],
                                    [next_dep[0], next_dep[1]],
                                    [next_arr[0], next_arr[1]])
    inblock_cons_check = check_inblock_cons(time_pair_for_inblock_check,
                                            config)
    check_f = bksta_cons_check + inblock_cons_check
    return check_f


# ================================================================
# check all trains according to future constraints (cross, consecutive, station and section constraints)
# with respect to arrival time in the current block and departure/potential arrival time in the next block.
# return an array about constraints types and trains id.
# ================================================================
def check_future_cons(times, config):
    if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = times
    n_agents = config.numT

    conflict_trains = []
    conflict_types = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # if i == 24 and j == 57:
            #     a = 1
            times_pair_for_inblock_check = ([if_stop_pre[i], if_stop_pre[j]],
                                            [if_stop_next[i], if_stop_next[j]],
                                            [pre_arr[i], pre_arr[j]],
                                            [pre_dep[i], pre_dep[j]],
                                            [pre_arr[i], pre_arr[j]])
            inblock_cons_check = check_inblock_cons(times_pair_for_inblock_check,
                                                    config)
            if len(inblock_cons_check) > 0:
                raise AssertionError(f'current in-block constraints happened of trains {i} and {j}')
            else:
                # check backward-station constraints in the next block
                times_pair_for_future_check = ([if_stop_next[i], if_stop_next[j]],
                                               [pre_arr[i], pre_arr[j]],
                                               [next_dep[i], next_dep[j]],
                                               [next_arr[i], next_arr[j]])
                conflict_type = _check_future_cons(times_pair_for_future_check, config)
                if len(conflict_type) > 0:
                    conflict_types.append(conflict_type)
                    conflict_trains.append([i, j])
    return np.array(conflict_trains), conflict_types


def find_nearest_train(train_id, pre_dep, pre_arr, config, mask_id=None):
    train_direction = np.array(config.direction)
    direction = train_direction[train_id]
    base_time = pre_arr[train_id]
    tid_arr = np.arange(len(pre_dep))
    mask_self = tid_arr != train_id
    if mask_id is not None:
        mask_self = np.logical_and(mask_self, tid_arr != mask_id)
    tids = tid_arr[mask_self]
    t_dirs = train_direction[tids]
    cmp_times = np.empty_like(tids, dtype=np.float64)
    if direction == 0:
        mask_up = (t_dirs == 1)
        cmp_times[mask_up] = pre_arr[tids[mask_up]]
        cmp_times[~mask_up] = pre_dep[tids[~mask_up]]
        mask_valid = cmp_times > base_time
        valid_ct = cmp_times[mask_valid]
        valid_tid = tids[mask_valid]
        if len(valid_ct) == 0:
            return None
        min_idx = np.argmin(valid_ct)
        # return (int(valid_tid[min_idx]), float(valid_ct[min_idx]))
        return int(valid_tid[min_idx])
    else:
        mask_up = (t_dirs == 1)
        cmp_times[mask_up] = pre_dep[tids[mask_up]]
        cmp_times[~mask_up] = pre_arr[tids[~mask_up]]
        mask_valid = cmp_times < base_time
        valid_ct = cmp_times[mask_valid]
        valid_tid = tids[mask_valid]
        if len(valid_ct) == 0:
            return None
        max_idx = np.argmax(valid_ct)
        return int(valid_tid[max_idx])


def find_bidirectional_nearest(train_id, next_dep, next_arr, config, mask_id=None):
    """
    无论输入列车为下行还是上行，同时查找其左侧和右侧最近的列车。
      - 右侧：候选上行取 pre_arr、候选下行取 pre_dep，
              取 > 目标列车 pre_arr 的最小者
      - 左侧：候选上行取 pre_dep、候选下行取 pre_arr，
              取 < 目标列车 pre_arr 的最大者
    :param train_id: 目标列车id
    :param next_dep: np.ndarray 各列车出发时刻
    :param next_arr: np.ndarray 各列车到达时刻
    :param config: 对象，config.direction 为各列车方向数组（0下行 1上行）
    :param mask_id: int|None 需要屏蔽的列车id（如冲突对象）
    :return: (left, right)
             left  = (left_id, left_cmp_time) 或 None（左侧无候选）
             right = (right_id, right_cmp_time) 或 None（右侧无候选）
    """
    train_direction = np.array(config.direction)
    base_time = next_arr[train_id]
    tid_arr = np.arange(len(next_dep))
    mask_self = tid_arr != train_id
    if mask_id is not None:
        mask_self = np.logical_and(mask_self, tid_arr != mask_id)
    tids = tid_arr[mask_self]
    t_dirs = train_direction[tids]
    # ============ 右侧最近============
    mask_up = (t_dirs == 1)
    right_times = np.empty_like(tids, dtype=np.float64)
    right_times[mask_up] = next_arr[tids[mask_up]]       # 候选上行：取arr
    right_times[~mask_up] = next_dep[tids[~mask_up]]     # 候选下行：取dep

    right_mask = right_times > base_time
    right_ct = right_times[right_mask]
    right_tid = tids[right_mask]
    if len(right_ct) == 0:
        right = None
    else:
        idx = np.argmin(right_ct)
        # right = (int(right_tid[idx]), float(right_ct[idx]))
        right = int(right_tid[idx])
    # ============ 左侧最近============
    left_times = np.empty_like(tids, dtype=np.float64)
    left_times[mask_up] = next_dep[tids[mask_up]]        # 候选上行：取dep
    left_times[~mask_up] = next_arr[tids[~mask_up]]      # 候选下行：取arr

    left_mask = left_times < base_time
    left_ct = left_times[left_mask]
    left_tid = tids[left_mask]
    if len(left_ct) == 0:
        left = None
    else:
        idx = np.argmax(left_ct)
        # left = (int(left_tid[idx]), float(left_ct[idx]))
        left = int(left_tid[idx])
    return left, right


def downstream_staHeadway_field(pre_arr, config):
    sta_field = []
    staHeadway = config.staHeadway
    direction = config.direction
    for i in range(len(pre_arr)):
        if direction[i] == 0:
            sta_field.append([pre_arr[i] - staHeadway, pre_arr[i] + staHeadway])
    return np.array(sta_field, dtype=np.int16)


def compute_adjust_field(_id, next_dep, next_arr, pre_arr, if_stop,
                         block_id, sta_field_for_uptrain, config, other_conflict_trains,
                         mask_id=None, if_adjusted=False):
    adjust_tw = []
    train_direction = config.direction
    direction = train_direction[_id]
    nearest_train = find_nearest_train(_id, next_dep, next_arr, config, mask_id)
    nearest_dep , nearest_arr = next_dep[nearest_train], next_arr[nearest_train]
    # if_stop[_id]=1说明该车已经被调整过或者即将被调整：
    # 如果已经被调整过，那么该车的next_dep已因疏解上次冲突而被调整，且可能不是最小停站时间导致的next_dep，此时tmp_dep只能是next_dep，
    # 并且可调整域需要计算上次停站导致的压缩量，为了维持之前已疏解冲突的关系，新的next_dep只能大于等于当前的next_dep，小于最大停站时间；
    # 如果没有被调整过，那么该车的next_dep还未计算，但pre_arr已经更新，可用pre_arr计算tmp_dep。
    # 如果一列车已经被调整过（停站时间不为0），那么其调整域也应当相对压缩。
    if if_adjusted or if_stop[_id]:
        if_adjusted = True
        tmp_dep = next_dep[_id]
        tmp_arr = next_arr[_id]
        if direction == 0:
            cmp_tmp_dep = pre_arr[_id] + config.ava_actions[1]
            cmp_tmp_arr = get_single_time(_id, cmp_tmp_dep, 1, None, 1,
                                          block_id, config)[0]
            cmp_dep = tmp_dep - cmp_tmp_dep
            cmp_arr = tmp_arr - cmp_tmp_arr
        else:
            cmp_tmp_dep = pre_arr[_id] - config.ava_actions[1]
            cmp_tmp_arr = get_single_time(_id, cmp_tmp_dep, 1, None, 1,
                                          block_id, config)[0]
            cmp_dep = cmp_tmp_dep - tmp_dep
            cmp_arr = cmp_tmp_arr - tmp_arr
    else:
        if direction == 0:
            tmp_dep = pre_arr[_id] + config.timeLossOfDc + config.ava_actions[1]
        else:
            tmp_dep = pre_arr[_id] - config.timeLossOfAc - config.ava_actions[1]
        tmp_arr = get_single_time(_id, tmp_dep, 1, None, 1,
                                  block_id, config)[0]
        cmp_tmp_arr = tmp_arr
        cmp_dep, cmp_arr = 0, 0
    # ====================== 如果和其他列车有冲突，则调整域为空 ======================
    # if _id in other_conflict_trains:
    #     tmp_times = (tmp_dep, tmp_arr)
    #     return np.array(adjust_tw, dtype=np.int16), tmp_times, nearest_train
    # ====================== 边界场景：无最近列车，直接用边界值 ======================
    if nearest_train is None:
        if direction == 0:
            # 下行右侧无车：min(边界, 最大到达时刻)为时间上限，可调整范围[最小到达时间, 时间上限] - 当前时间
            wall = min(config.timeZone, tmp_arr + (config.ava_actions[-1] - (config.ava_actions[1] + if_adjusted * cmp_arr)))
            adjust_tw = np.array([tmp_arr, wall], dtype=np.int16) - tmp_arr
        else:
            # 上行左侧无车：max(边界, 最大到达时刻)为时间下限，可调整范围[时间下限, 最小到达时间] - 当前时间
            wall = max(0, tmp_arr - (config.ava_actions[-1] - (config.ava_actions[1] + if_adjusted * cmp_arr)))
            _adjust_tw = np.array([wall, tmp_arr], dtype=np.int16) - tmp_arr
            _adjust_tw_dep = np.sort(tmp_dep + _adjust_tw)
            adjust_tw = np.sort(tmp_dep - subtract_time_windows(_adjust_tw_dep, sta_field_for_uptrain))
            # 更新调整后的时刻
            if len(adjust_tw) > 0 and np.min(adjust_tw) != 0:
                adjust_tw -= np.min(adjust_tw)
                tmp_dep -= np.min(adjust_tw)
                tmp_arr = get_single_time(_id, tmp_dep, 1, None, 1, block_id, config)[0]
        tmp_times = (tmp_dep, tmp_arr)
        return np.array(adjust_tw, dtype=np.int16), tmp_times, nearest_train
    # ====================== 正常场景：有最近列车，按间隔规则计算 ======================
    nearest_train_dir = train_direction[nearest_train]
    if direction == 0:
        if nearest_train_dir == 0:
            wall = (nearest_dep - config.cHeadwayWhenLaterPass)
        else:
            wall = nearest_arr - config.secHeadway
        wall = min(wall, tmp_arr + (config.ava_actions[-1] - (config.ava_actions[1] + if_adjusted * cmp_arr)))
        if tmp_arr <= wall:
            adjust_tw = np.array([tmp_arr, wall], dtype=np.int16) - tmp_arr
    else:
        if nearest_train_dir == 0:
            wall = nearest_arr + config.secHeadway
        else:
            wall = nearest_dep + config.cHeadwayWhenLaterPass
        wall = max(wall, tmp_arr - (config.ava_actions[-1] - (config.ava_actions[1] + if_adjusted * cmp_arr)))
        if tmp_arr >= wall:
            _adjust_tw = np.array([wall, tmp_arr], dtype=np.int16)  - tmp_arr
            _adjust_tw_dep = np.sort(tmp_dep + _adjust_tw)
            adjust_tw = np.sort(tmp_dep - subtract_time_windows(_adjust_tw_dep, sta_field_for_uptrain))
        if len(adjust_tw) > 0 and np.min(adjust_tw) != 0:
            tmp_dep -= np.min(adjust_tw)
            adjust_tw -= np.min(adjust_tw)
            tmp_arr = get_single_time(_id, tmp_dep, 1, None, 1, block_id, config)[0]
    tmp_times = (tmp_dep, tmp_arr)
    return np.array(adjust_tw, dtype=np.int16), tmp_times, nearest_train


def subtract_time_windows(a, b):
    a_s, a_e = a[0], a[1]
    overlap_mask = (b[:, 0] < a_e) & (b[:, 1] > a_s)
    overlap_b = b[overlap_mask]
    if len(overlap_b) == 0:
        return a.reshape(1, 2).copy()
    overlap_b = overlap_b[np.argsort(overlap_b[:, 0])]
    ov_s = np.maximum(overlap_b[:, 0], a_s)
    ov_e = np.minimum(overlap_b[:, 1], a_e)
    n_ov = len(ov_s)
    split_points = np.empty(2 * n_ov + 2, dtype=a.dtype)
    split_points[0] = a_s
    split_points[1:-1:2] = ov_s
    split_points[2:-1:2] = ov_e
    split_points[-1] = a_e
    candidates = split_points.reshape(-1, 2)
    result = candidates[candidates[:, 0] < candidates[:, 1]]
    return result


def find_ge_in_intervals(x: float, a: np.ndarray) -> float:
    """
    判断标量x是否落在任意区间内，在则返回x；否则返回大于x且最近的区间元素
    :param x: 输入标量
    :param a: shape=(n, 2) 的二维数组，每行代表一个闭区间 [start, end]，需满足 start <= end
    :return: 标量结果；无大于x的元素时返回 np.nan
    """
    a = np.asarray(a)
    # 1. 判断x是否在任意闭区间内
    in_any = np.any((a[:, 0] <= x) & (x <= a[:, 1]))
    if in_any:
        return x
    # 2. 不在区间内：找大于x的最小元素（等价于最小的区间左端点）
    greater_starts = a[a[:, 0] > x, 0]
    if greater_starts.size == 0:
        return False
    return float(np.min(greater_starts))


def find_le_in_intervals(x: float, a: np.ndarray) -> float:
    """
    判断标量x是否落在任意区间内，在则返回x；否则返回小于x且距离x最近的区间元素
    :param x: 输入标量
    :param a: shape=(n, 2) 的二维数组，每行代表一个闭区间 [start, end]，需满足 start <= end
    :return: 标量结果；无小于x的元素时返回 np.nan
    """
    a = np.asarray(a)
    # 1. 判断x是否在任意闭区间内
    in_any = np.any((a[:, 0] <= x) & (x <= a[:, 1]))
    if in_any:
        return x
    # 2. 不在区间内：找小于x的最大元素（等价于最大的区间右端点）
    less_ends = a[a[:, 1] < x, 1]
    if less_ends.size == 0:
        return np.nan
    return float(np.max(less_ends))


# ================================================================
# repair crossover and fd_sec conflicts
# ================================================================
def repair_cross_conflict(conflict_idx, times, next_block_id, config,
                          other_conflicts_ix, changed_idx, random_prob):
    success = True
    train_direction = config.direction
    if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = times
    i, j = conflict_idx[0], conflict_idx[1]
    former_train = i if next_dep[i] < next_dep[j] else j
    later_train = i if next_dep[i] > next_dep[j] else j
    former_dir = train_direction[former_train]
    later_dir = train_direction[later_train]
    sta_field_for_uptrain = downstream_staHeadway_field(pre_arr, config)
    infos = [i, j]
    assert former_dir == 0 and later_dir == 1, (f'Mistaken train directions in the cross/fd_sec conflict'
                                                f' {conflict_idx[0]}'
                                                f' and {conflict_idx[1]}')
    # adjust field
    former_ad, tmp_former_times, former_nearest_train = (
        compute_adjust_field(former_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                             sta_field_for_uptrain, config, other_conflicts_ix, mask_id=later_train))
    later_ad, tmp_later_times, later_nearest_train = (
        compute_adjust_field(later_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                             sta_field_for_uptrain, config, other_conflicts_ix,  mask_id=former_train))
    tmp_former_dep, tmp_former_arr = tmp_former_times # dep/arr time when a train is shifted to stop
    tmp_later_dep, tmp_later_arr = tmp_later_times # dep/arr time when a train is shifted to stop
    if former_nearest_train is not None:
        former_wt_dir = train_direction[former_nearest_train]
        # assert former_wt_dir == 1, "wrong former wall train direction"
    else:
        former_wt_dir = None
    if later_nearest_train is not None:
        later_wt_dir = train_direction[later_nearest_train]
        # assert later_wt_dir == 0, "wrong later wall train direction"
    else:
        later_wt_dir = None
    if len(former_ad):
        max_former_ad, min_former_ad = np.max(former_ad), np.min(former_ad)
        # assert min_former_ad == 0, 'Error for former_ad'
        max_delay_former_dep = tmp_former_dep + max_former_ad
        max_delay_former_arr = tmp_former_arr + max_former_ad
    if len(later_ad):
        max_later_ad, min_later_ad = np.max(later_ad), np.min(later_ad)
        # assert min_later_ad == 0, 'Error for later_ad'
        max_delay_later_dep = tmp_later_dep - max_later_ad
        max_delay_later_arr = tmp_later_arr - max_later_ad
    min_delay_former_dep, min_delay_former_arr = tmp_former_dep, tmp_former_arr
    min_delay_later_dep, min_delay_later_arr = tmp_later_dep, tmp_later_arr
    # -------------------- 嵌套辅助函数 --------------------
    def _refresh_former_ad():
        nonlocal former_ad, tmp_former_times, former_nearest_train
        nonlocal max_former_ad, min_former_ad, max_delay_former_dep, max_delay_former_arr
        nonlocal min_delay_former_dep, min_delay_former_arr, tmp_former_dep, tmp_former_arr

        former_ad, tmp_former_times, former_nearest_train = (
            compute_adjust_field(former_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                                 sta_field_for_uptrain, config, other_conflicts_ix, mask_id=later_train))
        tmp_former_dep, tmp_former_arr = tmp_former_times
        if len(former_ad):
            max_former_ad, min_former_ad = np.max(former_ad), np.min(former_ad)
            # assert min_former_ad == 0, 'Error for former_ad'
            max_delay_former_dep = tmp_former_dep + max_former_ad
            max_delay_former_arr = tmp_former_arr + max_former_ad
        else:
            max_former_ad = 0
            min_former_ad = 0
            max_delay_former_dep = tmp_former_dep
            max_delay_former_arr = tmp_former_arr
        min_delay_former_dep, min_delay_former_arr = tmp_former_dep, tmp_former_arr

    def _refresh_later_ad():
        """下行pre_arr变化后，重新计算站场域与上行列车可调整域及所有派生变量"""
        nonlocal sta_field_for_uptrain, later_ad, tmp_later_times, later_nearest_train
        nonlocal max_later_ad, min_later_ad, max_delay_later_dep, max_delay_later_arr
        nonlocal min_delay_later_dep, min_delay_later_arr, tmp_later_dep, tmp_later_arr
        # 1. 基于最新pre_arr重算站场间隔域
        sta_field_for_uptrain = downstream_staHeadway_field(pre_arr, config)
        # 2. 重算上行列车可调整域
        later_ad, tmp_later_times, later_nearest_train = compute_adjust_field(
            later_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
            sta_field_for_uptrain, config, other_conflicts_ix, mask_id=former_train)

        tmp_later_dep, tmp_later_arr = tmp_later_times

        # 3. 更新派生变量
        if len(later_ad):
            max_later_ad, min_later_ad = np.max(later_ad), np.min(later_ad)
            # assert min_later_ad == 0, 'Error for later_ad'
            max_delay_later_dep = tmp_later_dep - max_later_ad
            max_delay_later_arr = tmp_later_arr - max_later_ad
        else:
            max_later_ad = 0
            min_later_ad = 0
            max_delay_later_dep = tmp_later_dep
            max_delay_later_arr = tmp_later_arr
        min_delay_later_dep, min_delay_later_arr = tmp_later_dep, tmp_later_arr

    def _set_single_stop(train_idx, direction):
        """设置单列车停站并调整到达时间"""
        if if_stop_next[train_idx] == 0:
            if_stop_next[train_idx] = 1
            if direction == 0:  # 下行列车
                pre_arr[train_idx] += config.timeLossOfDc
                next_dep[train_idx] = pre_arr[train_idx] + config.ava_actions[1]
            else:  # 上行列车
                pre_arr[train_idx] -= config.timeLossOfAc
                next_dep[train_idx] = pre_arr[train_idx] - config.ava_actions[1]
            next_arr[train_idx] = get_single_time(train_idx, next_dep[train_idx], 1, None,
                                                  1, next_block_id, config)[0]
            if train_idx not in changed_idx:
                changed_idx.append(train_idx)

    def _set_both_stop():
        """设置两列车均停站"""
        _set_single_stop(former_train, former_dir)
        _set_single_stop(later_train, later_dir)

    def _calc_former_arr(dep_time):
        """封装前车到达时间计算"""
        return get_single_time(former_train, dep_time, 1, None, 1, next_block_id, config)[0]

    def _calc_later_arr(dep_time):
        """封装后车到达时间计算"""
        return get_single_time(later_train, dep_time, 1, None, 1, next_block_id, config)[0]

    def _handle_boundary_conflict():
        """处理「停站也无法满足间隔」的边界冲突"""
        nonlocal success, infos, changed_idx
        # 前后墙车方向
        # 下行列车与墙车反向
        if former_wt_dir is not None and former_dir != former_wt_dir and if_stop_next[former_nearest_train] and \
                former_nearest_train not in changed_idx:
            former_train_can_be_adjust = True
        else:
            former_train_can_be_adjust = False

        if former_train_can_be_adjust:
            # 前墙车可调整，后墙车不可调整
            if_stop_next[former_nearest_train] = 0
            pre_arr[former_nearest_train] += config.timeLossOfAc
            next_dep[former_nearest_train] = pre_arr[former_nearest_train]
            next_arr[former_nearest_train] = get_single_time(former_nearest_train, next_dep[former_nearest_train], 0,
                                                             None, 1, next_block_id, config)[0]
            changed_idx.append(former_nearest_train)
            _refresh_former_ad()
            if former_ad.size > 0:
                if max_delay_former_dep > next_dep[later_train] and \
                        max_delay_former_dep - next_dep[later_train] >= config.secHeadway:
                    if not if_stop_next[former_train]:
                        _set_single_stop(former_train, former_dir)
                    next_dep[former_train] = min(max_delay_former_dep, next_dep[later_train] + config.secHeadway)
                    next_arr[former_train] = _calc_former_arr(next_dep[former_train])
                    if former_train not in changed_idx:
                        changed_idx.append(former_train)
        else:
            success = False


    def _former_stop_enough():
        """仅前车停站即可满足间隔的处理逻辑"""
        _set_single_stop(former_train, former_dir)
        next_dep[former_train] = next_dep[later_train] + config.secHeadway
        next_arr[former_train] = _calc_former_arr(next_dep[former_train])

    def _later_stop_enough():
        """仅后车停站即可满足间隔的处理逻辑"""
        _set_single_stop(later_train, later_dir)
        next_dep[later_train] = next_dep[former_train] - config.staHeadway
        next_arr[later_train] = _calc_later_arr(next_dep[later_train])
    # -------------------------------------------------------------------------
    has_former_adjust = former_ad.size != 0
    has_later_adjust = later_ad.size != 0

    if has_former_adjust and has_later_adjust:
        # ===== 修改：分阶段随机逻辑 =====
        former_feasible = max_delay_former_dep >= next_dep[later_train] + config.secHeadway
        later_feasible = max_delay_later_dep <= next_dep[former_train] - config.secHeadway

        if np.random.random() < random_prob and former_feasible and later_feasible:
            if np.random.random() >= 0.5:
                _former_stop_enough()   # 下行停站、上行通过
            elif np.random.random() < 0.5:
                _later_stop_enough()    # 下行通过、上行停站
        else:
            # 开关关闭 / 仅一侧可行：原确定性优先级逻辑（下行优先）
            if former_feasible and max_former_ad >= max_later_ad:
                _former_stop_enough()
            elif later_feasible and max_former_ad <= max_later_ad:
                _later_stop_enough()
            else:
                # 两种单侧停站均不可行：两车均需停站，原逻辑不变
                _set_single_stop(former_train, former_dir)
                _refresh_later_ad()
                if later_ad.size == 0:
                    _handle_boundary_conflict()
                else:
                    _set_single_stop(later_train, later_dir)
                    _refresh_former_ad()
                    if min_delay_former_dep > min_delay_later_dep and \
                            (min_delay_former_dep - min_delay_later_dep) >= config.secHeadway:
                        next_dep[former_train] = min_delay_former_dep
                        next_arr[former_train] = tmp_former_arr
                        next_dep[later_train] = min_delay_later_dep
                        next_arr[later_train] = tmp_later_arr
                        if former_train not in changed_idx:
                            changed_idx.append(former_train)
                        if later_train not in changed_idx:
                            changed_idx.append(later_train)
                    elif max_delay_former_dep > max_delay_later_dep and \
                            (max_delay_former_dep - max_delay_later_dep) >= config.secHeadway:
                        next_dep[former_train] = max_delay_former_dep
                        next_arr[former_train] = max_delay_former_arr
                        next_dep[later_train] = max_delay_later_dep
                        next_arr[later_train] = max_delay_later_arr
                        if former_train not in changed_idx:
                            changed_idx.append(former_train)
                        if later_train not in changed_idx:
                            changed_idx.append(later_train)
                    else:
                        _handle_boundary_conflict()

    elif has_former_adjust and not has_later_adjust:
        # 情况2：仅前车有停站调整空间
        if max_delay_former_dep >= next_dep[later_train] + config.secHeadway:
            _former_stop_enough()
        else:
            _handle_boundary_conflict()
            # infos.append(later_nearest_train)
            # return False, infos, changed_idx
    elif not has_former_adjust and has_later_adjust:
        # 情况3：仅后车有停站调整空间
        if max_delay_later_dep <= next_dep[former_train] - config.secHeadway:
            _later_stop_enough()
        else:
            _handle_boundary_conflict()
            # infos.append(former_nearest_train)
            # return False, infos, changed_idx
    else:
        # 情况4：两车均无停站调整空间
        _handle_boundary_conflict()
    times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
    if success:
        return True, times, changed_idx
    else:
        return False, infos, changed_idx


# ================================================================
# repair consecutive conflicts
# ================================================================
def repair_consec_conflict(conflict_idx, times, next_block_id, config,
                           other_conflicts_ix):
    train_direction = config.direction
    if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = times
    i, j = conflict_idx[0], conflict_idx[1]
    former_train = i if next_dep[i] < next_dep[j] else j
    later_train = i if next_dep[i] > next_dep[j] else j
    former_dir = train_direction[former_train]
    later_dir = train_direction[later_train]
    infos = [i, j]
    assert former_dir == later_dir, (f'Mistaken train directions in the consec conflict {conflict_idx[0]}'
                                                f' and {conflict_idx[1]}')
    sta_field_for_uptrain = downstream_staHeadway_field(pre_arr, config)
    if former_dir == 0:
        # 下行，延迟后车
        later_ad, tmp_later_times, later_nearest_train = (
            compute_adjust_field(later_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                                 sta_field_for_uptrain, config, other_conflicts_ix))
        min_tmp_later_dep, min_tmp_later_arr = tmp_later_times
        if later_ad.size != 0:
            max_tmp_later_dep = min_tmp_later_dep + np.max(later_ad)
            if max_tmp_later_dep > next_arr[former_train] and \
                max_tmp_later_dep - next_arr[former_train] >= config.cHeadwayWhenLaterPass:
                if if_stop_next[later_train] == 0:
                    if_stop_next[later_train] = 1
                    pre_arr[later_train] += config.timeLossOfDc
                next_dep[later_train] = max(next_arr[former_train] + config.cHeadwayWhenLaterPass, min_tmp_later_dep)
                next_arr[later_train] = get_single_time(later_train, next_dep[later_train], 1, None,
                                                        1, next_block_id, config)[0]
            else:
                if later_nearest_train is not None:
                    infos.append(later_nearest_train)
                return False, infos
        else:
            if later_nearest_train is not None:
                infos.append(later_nearest_train)
            return False, infos
    else:
        # 上行，延迟前车
        former_ad, tmp_former_times, former_nearest_train = (
            compute_adjust_field(former_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                                 sta_field_for_uptrain, config, other_conflicts_ix))
        min_tmp_former_dep, min_tmp_former_arr = tmp_former_times
        # 如果上行列车的可调整域为0，返回失败；
        # 如果可调整域的SIZE不为0，说明至少可以最小停站，进一步检测以最大可调整量停站是否满足连发约束，如果满足则可疏解；否则返回失败。
        if former_ad.size != 0:
            max_tmp_former_dep = min_tmp_former_dep - np.max(former_ad)
            max_tmp_former_arr = min_tmp_former_arr - np.max(former_ad)
            if max_tmp_former_dep < next_arr[later_train] and \
                next_arr[later_train] - max_tmp_former_dep >= config.cHeadwayWhenLaterPass:
                if if_stop_next[former_train] == 0:
                    if_stop_next[former_train] = 1
                    pre_arr[former_train] -= config.timeLossOfAc
                next_dep[former_train] = min(next_arr[later_train] - config.cHeadwayWhenLaterPass, min_tmp_former_dep)
                next_arr[former_train] = get_single_time(former_train, next_dep[former_train], 1, None,
                                                        1, next_block_id, config)[0]
            else:
                if former_nearest_train is not None:
                    infos.append(former_nearest_train)
                return False, infos
        else:
            if former_nearest_train is not None:
                infos.append(former_nearest_train)
            return False, infos
    times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
    return True, times


# ================================================================
# repair station conflict
# ================================================================
def repair_sta_conflict(conflict_idx, times, next_block_id, config,
                           other_conflicts_ix, changed_idx):
    # 车站冲突发生在前车上行、后车下行交会场景，当前车停站时间不足时，其next_dep与下行方向后车pre_arr发生冲突。
    # 优先检测前车左侧调整域，如果可以延迟前车停站时间满足staHeadway，则如此；
    # 否则检测前车右侧调整域（屏蔽冲突列车id，因为将停站变为0必然不会和冲突列车发生其他冲突），如果右侧调整域较大，则将停站改为0。
    train_direction = config.direction
    if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = times
    i, j = conflict_idx[0], conflict_idx[1]
    former_train = i if next_dep[i] < next_dep[j] else j
    later_train = i if next_dep[i] > next_dep[j] else j
    former_dir = train_direction[former_train]
    later_dir = train_direction[later_train]
    assert if_stop_next[former_train], "upstream train not stop"
    assert former_dir == 1 and later_dir == 0, \
        f'Mistaken train directions in the sta conflict {conflict_idx[0]} and {conflict_idx[1]}'
    sta_field_for_uptrain = downstream_staHeadway_field(pre_arr, config)
    left_wall, right_wall = find_bidirectional_nearest(former_train, next_dep, next_arr, config, mask_id=later_train)
    former_ad, tmp_former_times, former_nearest_train = \
        compute_adjust_field(former_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                             sta_field_for_uptrain, config, other_conflicts_ix, if_adjusted=True)
    later_ad, tmp_later_times, later_nearest_train = \
        compute_adjust_field(later_train, next_dep, next_arr, pre_arr, if_stop_next, next_block_id,
                             sta_field_for_uptrain, config, other_conflicts_ix)
    min_tmp_former_dep, min_tmp_former_arr = tmp_former_times
    assert left_wall == former_nearest_train, "left_wall error"
    infos = [i, j]
    if former_ad.size != 0:
        max_tmp_former_dep = min_tmp_former_dep - np.max(former_ad)
        if max_tmp_former_dep < pre_arr[later_train] and \
                pre_arr[later_train] - max_tmp_former_dep >= config.staHeadway:
            assert if_stop_next[former_train], "upstream train not stop"
            next_dep[former_train] = pre_arr[later_train] - config.staHeadway
            next_arr[former_train] = get_single_time(former_train, next_dep[former_train], 1, None,
                                                     1, next_block_id, config)[0]
            times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
            return True, times, changed_idx
    else:
        if left_wall is not None:
            if left_wall not in changed_idx and if_stop_next[left_wall] and train_direction[left_wall] == 0 and \
                    pre_arr[former_train] - config.ava_actions[-1] < next_arr[left_wall] + config.secHeadway:
                if train_direction[left_wall] == 0:
                    pre_arr[left_wall] -= config.timeLossOfDc
                else:
                    pre_arr[left_wall] += config.timeLossOfAc
                next_dep[left_wall] = pre_arr[left_wall]
                next_arr[left_wall] = get_single_time(left_wall, next_dep[left_wall], 0, None,
                                                      1, next_block_id, config)[0]
                times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
                changed_idx.append(left_wall)
                return True, times, changed_idx
        # 检测前车是否可以转换为通过
        if former_train not in changed_idx and if_stop_next[former_train]:
            former_train_can_pass = True
        else:
            former_train_can_pass = False
        # 检测后车是否有空间停车与前车满足会车间隔约束
        former_train_no_secConflict =False
        if later_ad.size > 0:
            if (next_dep[later_train] + np.max(later_ad)) - (pre_arr[former_train] + config.timeLossOfAc) >= \
                config.secHeadway:
                former_train_no_secConflict = True
        # 当且仅当前车的右墙车==后墙车（后墙车没有插在两车之间）；和前车可以转换为通过；和后车有空间与前车满足会车间隔约束时：前车可以转换为通过
        if right_wall == later_nearest_train and former_train_can_pass and former_train_no_secConflict:
            if_stop_next[former_train] = 0
            pre_arr[former_train] += config.timeLossOfAc
            next_dep[former_train] = pre_arr[former_train]
            next_arr[former_train] = get_single_time(former_train, next_dep[former_train], 0, None,
                                                     1, next_block_id, config)[0]
            times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
            changed_idx.append(former_train)
            return True, times, changed_idx
    return False, infos, changed_idx


# ================================================================
# repair future conflicts
# ================================================================
def repair_future_conflicts(times, actions,
                            block_id, config,
                            test_mode=False, random_prob=0.0):
    if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = times
    next_block_id = block_id + 1
    if test_mode:
        print(next_block_id)
    if next_block_id == 1:
        a = 1
    _actions = actions.copy().astype(np.int32)
    changed_idx = []
    train_directions = config.direction

    # if test_mode:
    #     time_slot = np.vstack((pre_dep, pre_arr, next_dep, next_arr))
    #     # print('time_slot:\n', time_slot)
    #     Draw(config, time_slot, next_block_id - 1)

    for _ in range(1000):

        if test_mode:
            time_slot = np.vstack((pre_dep, pre_arr, next_dep, next_arr))
            # print('time_slot:\n', time_slot)
            Draw(config, time_slot, next_block_id - 1)

        conflict_trains, conflict_types = check_future_cons(times, config)
        # conflict_trains = np.array(conflict_trains)
        if len(conflict_types) != 0:
            for k in range(len(conflict_trains)):
                i, j = conflict_trains[k]
                if i == 3 and j == 8:
                    a = 1

                _oc_msk = (conflict_trains == np.array(conflict_trains[k])).all(axis=1)
                other_conflict_idx = conflict_trains[~_oc_msk]
                # if len(conflict_types[k]) == 1 and 'sta' in conflict_types[k]:
                #     indicator, infos, changed_idx = repair_sta_conflict(conflict_trains[k], times, next_block_id, config,
                #                                 other_conflict_idx, changed_idx)
                #     if indicator == False:
                #         _times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
                #         return _times, _actions, 'failure', infos
                #     else:
                #         if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = infos
                #     break

                if i not in changed_idx and if_stop_next[i]:
                    _actions[i] = 0
                    if_stop_next[i] = 0
                    if train_directions[i] == 0:
                        pre_arr[i] -= config.timeLossOfDc
                    else:
                        pre_arr[i] += config.timeLossOfAc
                    next_dep[i] = pre_arr[i]
                    next_arr[i] = get_single_time(i, next_dep[i], 0,
                                            None, 1, next_block_id, config)[0]
                    changed_idx.append(i)

                if j not in changed_idx and if_stop_next[j]:
                    _actions[j] = 0
                    if_stop_next[j] = 0
                    if train_directions[j] == 0:
                        pre_arr[j] -= config.timeLossOfDc
                    else:
                        pre_arr[j] += config.timeLossOfAc
                    next_dep[j] = pre_arr[j]
                    next_arr[j] = get_single_time(j, next_dep[j], 0,
                                                  None, 1, next_block_id, config)[0]
                    changed_idx.append(j)

                # if i in changed_idx and j in changed_idx:
                #     continue

                _times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
                #
                if test_mode:
                    time_slot = np.vstack((pre_dep, pre_arr, next_dep, next_arr))
                    # print('time_slot:\n', time_slot)
                    Draw(config, time_slot, next_block_id - 1)

                time_pair = ([if_stop_next[i], if_stop_next[j]],
                             [pre_arr[i], pre_arr[j]],
                             [next_dep[i], next_dep[j]],
                             [next_arr[i], next_arr[j]])
                conflict_type = _check_future_cons(time_pair, config)
                if len(conflict_type) != 0:
                    if 'cross' in conflict_type or 'fd_sec' in conflict_type:
                        indicator, infos, changed_idx = repair_cross_conflict(conflict_trains[k], _times, next_block_id, config,
                                                      other_conflict_idx, changed_idx, random_prob=random_prob)
                        if indicator == False:
                            return _times, _actions, 'failure', infos
                        else:
                            if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = infos
                            break
                    elif 'consec' in conflict_type or 'overtaking' in conflict_type:
                        indicator, infos = repair_consec_conflict(conflict_trains[k], _times, next_block_id, config,
                                                       other_conflict_idx)
                        if indicator == False:
                            return _times, _actions, 'failure', infos
                        else:
                            if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr = infos
                            break
                    else:
                        return _times, _actions, 'failure', infos
                        # print(f'conflict_trains:\n{conflict_trains}')
                        # print(f'conflict_types:\n{conflict_types}')
                        # print(f'conflict_block: {next_block_id}')
                        # raise AssertionError('Error conflict type')
                else:
                    break
        else:
            _times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
            _actions = np.abs(next_dep - pre_arr)
            return _times, _actions, 'success', []
    # _times = (if_stop_pre, if_stop_next, pre_dep, pre_arr, next_dep, next_arr)
    # _actions = np.abs(next_dep - pre_arr)
    # return _times, _actions, 'failure',

# ================================================================
# check all trains according to all constraints (station and section constraints)
# return an array about which train disobey the constraints
# ================================================================
def check_all_cons(times, config):
    ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr = times
    numT = config.numT
    timeZone = config.timeZone
    check = np.array([0 for _ in range(numT)])

    # whether exceed the scale of timetable
    for i in range(numT):
        if pre_arr[i] < 0 or pre_arr[i] > timeZone:
            check[i] += 1
    if next_arr is not None:
        for i in range(numT):
            if next_arr[i] < 0 or next_arr[i] > timeZone:
                check[i] += 1
     # assert ifExceed == 0,'some trains exceed the scale of timetable'

    for i in range(numT - 1):
        for j in range(i + 1, numT):
            # check train order
            formalT = i if pre_dep[i] < pre_dep[j] else j
            laterT = i if pre_dep[i] > pre_dep[j] else j
            # formD: former train's departure time
            # formA: former train's arrival time
            # laterD: later train's departure time
            # laterA: later train's arrival time
            # next_formD: former train's departure time in next block
            # next_laterD: later train's departure time in next block
            formD, formA, laterD, laterA, next_formD, next_laterD = \
                pre_dep[formalT], pre_arr[formalT], pre_dep[laterT], pre_arr[laterT], next_dep[formalT], next_dep[
                    laterT]
            LaterStop = [ifStopPre[laterT], ifStopNext[laterT]]
            time, stopLater = [formD, formA, laterD, laterA, next_formD, next_laterD], LaterStop

            check[i] += check_cross_cons(time)
            check[j] += check_cross_cons(time)
            check[i] += check_consec_cons(time, stopLater)
            check[j] += check_consec_cons(time, stopLater)
            check[i] += check_stat_and_sect_cons(time)
            check[j] += check_stat_and_sect_cons(time)

            if next_arr is not None:
                formalT = i if next_dep[i] < next_dep[j] else j
                laterT = i if next_dep[i] > next_dep[j] else j
                next_formD, next_laterD, formDestinationA, LaterDestinationA = \
                    next_dep[formalT], next_dep[laterT], next_arr[formalT], next_arr[laterT]
                # last block check constraints
                time, stopLater = [next_formD, formDestinationA, next_laterD, LaterDestinationA, None, None], \
                    [ifStopNext[laterT], 1]
                check[i] += check_cross_cons(time)
                check[j] += check_cross_cons(time)
                check[i] += check_consec_cons(time, stopLater)
                check[j] += check_consec_cons(time, stopLater)
                check[i] += check_stat_and_sect_cons(time)
                check[j] += check_stat_and_sect_cons(time)
    return check


# ================================================================
# check all trains according to all constraints (station and section constraints)
# return an array about which train disobey the constraints
# ================================================================
def get_available_actions(n_agents, optional_actions, stop_plan):

    available_actions = np.ones((n_agents, len(optional_actions)))

    if stop_plan is not None:
        for i in range(n_agents):
            # if stop_plan[i] == 0:
            #     available_actions[i, 1:] = 0
            # else:
            #     available_actions[i, 0] = 0
            if stop_plan[i]:
                available_actions[i, 0] = 0

    return available_actions




# if __name__ == "__main__":



