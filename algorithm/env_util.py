import numpy as np


# ================================================================
# read the excel
# ================================================================
# def e2m(path):
#     table = xlrd.open_workbook(path).sheets()[0]  # get the first sheet from the Excel file
#     row = table.nrows
#     col = table.ncols
#     datamatrix = np.zeros((row, col))  # generate a matrix (@row * @col) filled with 0
#     for x in range(col):
#         cols = np.matrix(table.col_values(x))
#         datamatrix[:, x] = cols
#     return datamatrix


# ================================================================
# get the binary about whether the train stop at previous station and later station
# as well as departure time and arrive time from @STATES and @ACTIONS
# ================================================================
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


# def get_time(states, actions, current_station, timeZone, numT, numB, downRunTime, upRunTime):
def get_time(states, actions, current_station, config):

    numT = config.numT
    numB = config.numB
    timeZone = config.timeZone

    pre_dep = [np.round(states[3 * i] * timeZone) for i in range(numT)]  # 在上一车站（当前区间的后方站）的到达时刻
    direction = [states[3 * i + 1] for i in range(numT)]  # 列车运行方向
    ifStopPre = [states[3 * i + 2] for i in range(numT)]  # 在上一车站（当前区间的后方站）是否停车

    pre_arr = [0 for _ in range(numT)]  # 在当前车站（当前区间的前方站）的到达时刻
    next_dep = [0 for _ in range(numT)]  # 在当前车站（下一区间的后方站）的出发时刻
    ifStopNext = [1 if actions[i] > 0 else 0 for i in range(numT)]  # 在当前车站（下一区间的后方站）是否停车
    next_arr = None

    location = current_station  # 当前车站编号
    n_section = int(location) - 1  # 当前区间编号

    for i in range(numT):

        pre_arr[i], next_dep[i] = get_single_time(i, pre_dep[i], ifStopPre[i], actions[i],
                                                  ifStopNext[i], n_section, config)

        # if direction[i] == 0:  # 下行列车
        #     pre_arr[i] = pre_dep[i] + ifStopPre[i] * timeLossOfAc + \
        #                  ifStopNext[i] * timeLossOfDc + downRunTime[n_section]
        #     next_dep[i] = pre_arr[i] + actions[i]
        # else:  # 上行列车
        #     pre_arr[i] = pre_dep[i] - ifStopPre[i] * timeLossOfDc - \
        #                  ifStopNext[i] * timeLossOfAc - upRunTime[n_section]
        #     next_dep[i] = pre_arr[i] - actions[i]

    if n_section == numB - 2:  # 终到区间的前一个区间，将两个区间统一考虑
        next_arr = [0 for _ in range(numT)]  # 在下一车站（下一区间的前方站）即终到站的到达时刻
        ifStopPre = ifStopNext
        ifStopNext = [1 for _ in range(numT)]
        for i in range(numT):
            next_arr[i], _ = get_single_time(i, next_dep[i], ifStopPre[i], None, ifStopNext[i],
                                             n_section=-1, config=config)

            # if direction[i] == 0:
            #     next_arr[i] = next_dep[i] + ifStopNext[i] * timeLossOfAc + \
            #                   timeLossOfDc + downRunTime[-1]
            # else:
            #     next_arr[i] = next_dep[i] - ifStopNext[i] * timeLossOfDc - \
            #                   timeLossOfAc - downRunTime[-1]

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

    p1 = {"x": formD, "y": 0}
    p2 = {"x": formA, "y": 1}
    p3 = {"x": laterD, "y": 0}
    p4 = {"x": laterA, "y": 1}

    def cross(_p1, _p2, _p3):  # 跨立实验
        x1 = _p2["x"] - _p1["x"]
        y1 = _p2["y"] - _p1["y"]
        x2 = _p3["x"] - _p1["x"]
        y2 = _p3["y"] - _p1["y"]
        return x1 * y2 - x2 * y1

    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(p1["x"], p2["x"]) >= min(p3["x"], p4["x"])  # 矩形1最右端大于矩形2最左端
            and max(p3["x"], p4["x"]) >= min(p1["x"], p2["x"])  # 矩形2最右端大于矩形最左端
            and max(p1["y"], p2["y"]) >= min(p3["y"], p4["y"])  # 矩形1最高端大于矩形最低端
            and max(p3["y"], p4["y"]) >= min(p1["y"], p2["y"])):  # 矩形2最高端大于矩形最低端

        # 若通过快速排斥则进行跨立实验
        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return D


# ================================================================
# check all trains according to station and section constraints
# return the number of trains who disobey any constraint (i.e. sHeadway, cHeadway or crossover constraint)
# ================================================================
def check_stat_and_sect_cons(time):
    """
    检查不同时到达时间间隔和会车时间间隔
    time: 时刻向量
    formD: 前发车在当前区间后方站的出发时刻
    formA: 前发车在当前区间前方站的到达时刻
    next_formD: 前发车在下一区间后方站（即当前区间前方站）的出发时刻
    laterD: 后发车在当前区间后方站的出发时刻
    laterA: 后发车在当前区间前方站的到达时刻
    next_laterD: 后发车在下一区间后方站（即当前区间前方站）的出发时刻
    formDirection: 前发车的运行方向
    laterDirection: 后发车的运行方向
    """

    staHeadway = 2
    secHeadway = 2

    checkS = 0  # 违背车站约束的列车数量

    formD, formA, laterD, laterA, next_formD, next_laterD = \
        time[0], time[1], time[2], time[3], time[4], time[5]

    formDirection = 0 if formD < formA else 1
    laterDirection = 0 if laterD < laterA else 1

    if formDirection == 0 and laterDirection == 1:
        # 前车为下行列车，后车为上行列车，
        if next_laterD is None and next_formD is None:
            # 终到区间只检查终到时刻的会车间隔
            if laterA - formA < secHeadway:
                checkS += 1
        else:
            # 非终到区间
            # 首先确定两列车是否为相邻关系
            # close_relation = 1 if next_formD > next_laterD else 0
            # if close_relation == 1:
            #     if next_laterD != laterA and next_formD != formA:
            #         checkS += 1
            #         # pass
            #     elif next_formD == formA and next_laterD != laterA:
            #         if formA - next_laterD < staHeadway:
            #             checkS += 1
            #         if laterD - formA < secHeadway:
            #             checkS += 1
            #     elif next_laterD == laterA and next_formD != formA:
            #         if next_laterD - formA < staHeadway:
            #             checkS += 1
            #         if next_formD - next_laterD < secHeadway:
            #             checkS += 1
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

    cHeadwayWhenLaterStop = 4  # 后车在区间前方站停站时的连发时间间隔
    cHeadwayWhenLaterPass = 4  # 后车在区间前方站不停站时的连发时间间隔

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
# check neighbouring trains according to all constraints (cross, station and section constraints)
# with respect to potential departure and arrival time in the next section.
# return an array about which train disobey the constraints.
# ================================================================
def check_future_cons(times, config):

    timeZone = config.timeZone

    check_f = 0
    if_stop_pre, if_stop_next, pre_dep, pre_arr = times

    c_pre_dep = pre_dep[1]
    c_pre_arr = pre_arr[1]

    # get the order of both trains
    former_train = 0 if pre_dep[0] < pre_dep[1] else 1
    later_train = 0 if pre_dep[0] > pre_dep[1] else 1

    # check the time zone
    if c_pre_arr < 0 or c_pre_arr > timeZone:
        check_f = 1
    else:
        _times = (pre_dep[former_train], pre_arr[former_train], pre_dep[later_train], pre_arr[later_train])
        check_f = check_cross_cons(_times)
        if check_f == 0:

            # get directions of both trains
            c_direction = 0 if c_pre_dep < c_pre_arr else 1
            n_direction = 0 if pre_dep[0] < pre_arr[0] else 1

            if c_direction == 0 and n_direction == 0:
                # both trains are downstream, check consecutive constraints
                # stopDownLater = stopLater[0]
                # stopUpLater = stopLater[1]
                # formD, formA, laterD, laterA = time[0], time[1], time[2], time[3]
                _time = (pre_dep[former_train], pre_arr[former_train],
                         pre_dep[later_train], pre_arr[later_train])
                _stopLater = (if_stop_pre[later_train], -1)
                check_f = check_consec_cons(_time, _stopLater)

            elif c_direction == 1 and n_direction == 1:
                _time = (pre_dep[former_train], pre_arr[former_train],
                         pre_dep[later_train], pre_arr[later_train])
                _stopLater = (-1, if_stop_next[later_train])
                check_f = check_consec_cons(_time, _stopLater)

            elif (former_train == 1 and (c_direction == 0 and n_direction == 1)) or \
                    (later_train == 1 and (c_direction == 1 and n_direction == 0)):
                if pre_arr[later_train] - pre_arr[former_train] < config.secHeadway:
                    check_f = 1

    return check_f


# ================================================================
# check all trains according to all constraints (station and section constraints)
# return an array about which train disobey the constraints
# ================================================================
def check_all_cons(times, timeZone, numT):
    ifStopPre, ifStopNext, pre_dep, pre_arr, next_dep, next_arr = times

    check = np.array([0 for _ in range(numT)])

    # whether exceed the scale of timetable
    # 检查到发时刻是否在运营时间范围内
    # for i in range(numT):
    #     if pre_arr[i] < 0 or pre_arr[i] > timeZone:
    #         check[i] += 1
    if next_arr is not None:
        for i in range(numT):
            if next_arr[i] < 0 or next_arr[i] > timeZone:
                check[i] += 1
    #  assert ifExceed == 0,'some trains exceed the scale of timetable'

    for i in range(numT - 1):
        for j in range(i + 1, numT):
            # 根据确定前、后车顺序
            formalT = i if pre_dep[i] < pre_dep[j] else j
            laterT = i if pre_dep[i] > pre_dep[j] else j

            # assert formalT != laterT, "请检查前、后车顺序!"

            # 获取前车和后车在当前区间的后方站出发时刻、前方站的到达时刻和出发时刻。
            formD, formA, laterD, laterA, next_formD, next_laterD = \
                pre_dep[formalT], pre_arr[formalT], pre_dep[laterT], pre_arr[laterT], next_dep[formalT], next_dep[
                    laterT]

            # 获取后车是否停站bool变量，以检查连发时间间隔约束
            LaterStop = [ifStopPre[laterT], ifStopNext[laterT]]

            time, stopLater = [formD, formA, laterD, laterA, next_formD, next_laterD], LaterStop
            check[i] += check_cross_cons(time)
            check[j] += check_cross_cons(time)
            check[i] += check_consec_cons(time, stopLater)
            check[j] += check_consec_cons(time, stopLater)
            check[i] += check_stat_and_sect_cons(time)
            check[j] += check_stat_and_sect_cons(time)

            # 如果到达倒数第二个区间，则获取前车和后车在下一区间（即终到区间）前方站的到达时刻（即终到时刻）。
            if next_arr is not None:
                # 重新确定列车在最后一个区间内的前、后车顺序
                formalT = i if next_dep[i] < next_dep[j] else j
                laterT = i if next_dep[i] > next_dep[j] else j
                next_formD, next_laterD, formDestinationA, LaterDestinationA = \
                    next_dep[formalT], next_dep[laterT], next_arr[formalT], next_arr[laterT]
                # 检查终到区间的约束条件满足情况
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
#     formD = 59
#     formA = 68
#     next_formD = 68
#     laterD = 87
#     laterA = 78
#     next_laterD = 67
#     time = (formD, formA, laterD, laterA, next_formD, next_laterD)
#     checkS = check_stat_and_sect_cons(time)
