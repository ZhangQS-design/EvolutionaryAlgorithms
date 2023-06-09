"""
@description: 辅助函数
"""
import time
import os

import numpy as np
import random
import copy

import pickle
import pandas as pd
import math

from EntityDefinition import TaskGroup, Task, SourceCenter, SubTask
from functools import lru_cache
from concurrent import futures

from Configuration import fog_compute_power_range, fog_trans_rate_range, fog_memory_range, v_compute_power_range, \
    v_trans_rate_range, v_memory_range, cloud_compute_delay, cloud_transmission_rate, cloud_memory, cloud_id, \
    task_num_range, p_cross, p_mutation, p_reverse, v2f_limit_distance, \
    v2v_limit_distance, fogs_location, sub_task_num_range, upload_size_range, \
    download_size_range, compute_cost_range, storage_cost_range, total_tasks_number, tasks_pool_path, tasks_table_path, \
    data, origin_source, vehicles_attrs, fogs_attrs, lam, uav_compute_power_range, uav_trans_rate_range, \
    uav_memory_range, uavs_location, uavs_attrs, data_uav, u2vf_limit_distance

import sys

sys.setrecursionlimit(10000)

total_request_task_num = 0
severed_task_num = 0


# 情况路径下所有文件
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
            del_file(c_path)
        else:  # 如果是一个文件那么直接删除
            os.remove(c_path)
    print('文件已经清空完成')


# 寻找辅助无人机位置
def search_auxiliary_uav_location(v_x, v_y, dispatch_time):
    selected_uav_x = -1
    selected_uav_y = -1
    selected_uav_id = 105
    dis_v_uav = 1000000000
    # for i in range(105, 111):
    # vss = data_uav.loc[dispatch_time]
    # vss = vss.set_index('uav_id')
    # u_x = vss.loc[i][0]
    # u_y = vss.loc[i][1]
    for v in range(6):
        uav_id = data_uav.loc[dispatch_time].iloc[v, 0]
        u_x = data_uav.loc[dispatch_time].iloc[v, 1]
        u_y = data_uav.loc[dispatch_time].iloc[v, 2]
        distance_vehicle2uav = math.sqrt((v_x - u_x) ** 2 + (v_y - u_y) ** 2)
        if distance_vehicle2uav < dis_v_uav:
            selected_uav_x = u_x
            selected_uav_y = u_y
            dis_v_uav = distance_vehicle2uav
            selected_uav_id = uav_id
        # print("selected_uav_x", selected_uav_x)
        # print("selected_uav_y", selected_uav_y)
        # print("selected_uav_id", selected_uav_id)
    return selected_uav_x, selected_uav_y, selected_uav_id


# 返回某一时间下某一位置的可通信节点集合
def compute_distance_and_save_usable_nodes(v_x, v_y, time_step):
    usable_nodes_set = []
    for each_v_idx in range(data.loc[time_step].shape[0]):
        each_vehicle_id = data.loc[time_step].iloc[each_v_idx, 0]
        each_vehicle_x = data.loc[time_step].iloc[each_v_idx, 1]
        each_vehicle_y = data.loc[time_step].iloc[each_v_idx, 2]
        distance2vehicle = math.sqrt((v_x - each_vehicle_x) ** 2 + (v_y - each_vehicle_y) ** 2)

        if distance2vehicle <= v2v_limit_distance:
            usable_nodes_set.append(each_vehicle_id)

    for each_f in fogs_location:
        each_fog_id = fogs_location[each_f][0]
        each_fog_x = fogs_location[each_f][1]
        each_fog_y = fogs_location[each_f][2]
        distance2fog = math.sqrt((v_x - each_fog_x) ** 2 + (v_y - each_fog_y) ** 2)

        if distance2fog <= v2f_limit_distance:
            usable_nodes_set.append(each_fog_id)

    usable_nodes_set.append(cloud_id)

    # 添加符合通信范围的无人机
    for each_uav_idx in range(data_uav.loc[time_step].shape[0]):
        each_uav_id = data_uav.loc[time_step].iloc[each_uav_idx, 0]
        each_uav_x = data_uav.loc[time_step].iloc[each_uav_idx, 1]
        each_uav_y = data_uav.loc[time_step].iloc[each_uav_idx, 2]
        distance2uav = math.sqrt((v_x - each_uav_x) ** 2 + (v_y - each_uav_y) ** 2)
        if distance2uav <= u2vf_limit_distance:
            usable_nodes_set.append(each_uav_id)
    return usable_nodes_set


# 判断某一时刻下请求方与服务方是否能保持通信
def calculate_two_vehicle_is_interact(delayed_time, request_v_id, response_v_id):
    request_f = 'request_data/time' + str(delayed_time) + '-vehicle' + str(request_v_id) + 'request.data'

    try:
        with open(request_f, 'rb') as f:
            request_task_group = pickle.load(f)
        sustain_space = request_task_group.communication_space
        if response_v_id in sustain_space:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


# 随机生成不同时刻下的任务表单
def construct_tasks_table_and_update_tasks_pool():
    tasks_table = np.zeros((total_tasks_number, sub_task_num_range[1] * 4 + 2))
    tasks_table[:, :] = None
    # print(tasks_table.shape)
    tasks_pool = []
    for i in range(total_tasks_number):

        i_identifier = i
        i_sub_tasks_num = random.randint(sub_task_num_range[0], sub_task_num_range[1])
        i_task_info_dict = dict()

        tasks_table[i, 0] = i_identifier
        tasks_table[i, 1] = i_sub_tasks_num

        for j in range(i_sub_tasks_num):
            us = 0.01 * random.randint(int(upload_size_range[0] * 100), int(upload_size_range[1] * 100))
            # ds = random.randint(download_size_range[0], download_size_range[1])
            ds = 0.01 * random.randint(int(upload_size_range[0] * 100), int(upload_size_range[1] * 100))
            cc = random.randint(compute_cost_range[0], compute_cost_range[1]) * pow(10, 9)
            sc = random.randint(storage_cost_range[0], storage_cost_range[1])
            tasks_table[i, 2 + j * 4:2 + (j + 1) * 4] = [us, ds, cc, sc]
            i_task_info_dict['sub_task_' + str(j)] = [us, ds, cc, sc]
        create_task = Task(timestamp=-404, vehicle_id=-404, uav_id=-404, task_order=-404, communication_space=[],
                           sub_tasks_num=i_sub_tasks_num, task_info_dict=i_task_info_dict)
        tasks_pool.append(create_task)

    sub_titles = ['task_identifier', 'sub_tasks_num']
    for x in range(sub_task_num_range[1]):
        sub_titles.append('sub_task_' + str(x) + '.up_size')
        sub_titles.append('sub_task_' + str(x) + '.down_size')
        sub_titles.append('sub_task_' + str(x) + '.compute_cost')
        sub_titles.append('sub_task_' + str(x) + '.storage_cost')
    # print(sub_titles)
    tasks_table_df = pd.DataFrame(tasks_table, index=None, columns=sub_titles)

    tasks_table_df.to_csv(tasks_table_path, index=False)
    fo = tasks_pool_path
    with open(fo, 'wb+') as f:
        pickle.dump(obj=tasks_pool, file=f)


# 预先抽取所有时间下的任务
# 预先抽取所有时间下的任务
# 将不同时刻下任务同步到该时刻下的车辆中
def tasks_release():

    path = "request_data"
    del_file(path)

    f1 = tasks_pool_path
    with open(f1, 'rb') as f:
        tasks_pool = pickle.load(f)

    repeat_filter = np.zeros((100, 300)) - 1

    for t in range(1, 301):
        total_vehicle_num_in_area = data.loc[t].shape[0]

        for v in range(total_vehicle_num_in_area):

            vehicle_id = data.loc[t].iloc[v, 0]
            vehicle_x = data.loc[t].iloc[v, 1]
            vehicle_y = data.loc[t].iloc[v, 2]
            # 搜索辅助无人机
            uav_x, uav_y, uav_id = search_auxiliary_uav_location(vehicle_x, vehicle_y, t)
            # vehicle_interval = int(vehicles_attrs.iloc[vehicle_id-1, 3])
            # vehicle_request_times = list(range(1, 301, vehicle_interval))
            vehicle_request_times = []
            send_request = 1
            request_time = 1
            while send_request:
                if request_time < 301:
                    request_interval = int(np.random.exponential(lam)) + 1
                    vehicle_request_times.append(request_time)
                    request_time += request_interval
                else:
                    send_request = 0
            # print(vehicle_request_times)

            if t in vehicle_request_times:
                # 车辆的通信空间
                # communication_space = \
                #     compute_distance_and_save_usable_nodes(v_x=vehicle_x, v_y=vehicle_y, time_step=t)
                # 无人机的通信空间
                communication_space = compute_distance_and_save_usable_nodes(v_x=uav_x, v_y=uav_y, time_step=t)
                t_v_task_num = random.randint(task_num_range[0], task_num_range[1])
                task_set = []
                count = 0
                for i in range(t_v_task_num):  # 确定了任务数量， 选择具体的任务
                    while True:
                        i_random_idx = random.randint(0, total_tasks_number - 1)
                        i_select = tasks_pool[i_random_idx]
                        if i_random_idx not in repeat_filter[vehicle_id, 0:t - 1]:
                            repeat_filter[vehicle_id, t - 1] = i_random_idx
                            count += 1

                            i_ready = copy.deepcopy(i_select)
                            i_ready.update_task_part_attr(timestamp=t, vehicle_id=vehicle_id,
                                                          uav_id=uav_id, task_order=count,
                                                          communication_space=copy.deepcopy(communication_space))

                            task_set.append(i_ready)
                            break

                tg = TaskGroup(timestamp=t, vehicle_id=vehicle_id, uav_id=uav_id, task_pool=task_set,
                               communication_space=copy.deepcopy(communication_space))
                print(tg.get_all_information())
                fv = 'request_data/time' + str(t) + '-vehicle' + str(vehicle_id) + 'request.data'
                print(fv)
                with open(fv, 'wb+') as f:
                    pickle.dump(obj=tg, file=f)


# 为车辆节点和雾节点生成属性：计算能力，传输速率， 存储空间
def generate_attribution_for_each_node():
    fogs_attr = np.zeros((4, 3))
    for ii in range(0, 4):
        fog_computing_power = random.randint(fog_compute_power_range[0], fog_compute_power_range[1]) * pow(10, 9)
        fog_transmission_rate = random.randint(fog_trans_rate_range[0], fog_trans_rate_range[1])
        fog_memory = random.randint(fog_memory_range[0], fog_memory_range[1])

        fogs_attr[ii, 0] = fog_computing_power
        fogs_attr[ii, 1] = fog_transmission_rate
        fogs_attr[ii, 2] = fog_memory

    df1 = pd.DataFrame(fogs_attr)
    df1.to_csv('data/fogs_attr.csv', header=False, index=False)

    vehicles_attr = np.zeros((100, 3))
    uavs_attr = np.zeros((6, 3))
    for jj in range(0, 100):
        vehicle_computing_power = random.randint(v_compute_power_range[0], v_compute_power_range[1]) * pow(10, 9)
        vehicle_transmission_rate = random.randint(v_trans_rate_range[0], v_trans_rate_range[1])
        vehicle_memory = random.randint(v_memory_range[0], v_memory_range[1])
        # vehicle_interval = int(np.random.exponential(lam)) + 1

        vehicles_attr[jj, 0] = vehicle_computing_power
        vehicles_attr[jj, 1] = vehicle_transmission_rate
        vehicles_attr[jj, 2] = vehicle_memory
        # vehicles_attr[jj, 3] = vehicle_interval
        # vehicles_attr[jj, 3] = vehicle_interval

    for kk in range(0, 6):
        # 无人机的计算、存储和通信资源
        uav_computing_power = random.randint(uav_compute_power_range[0], uav_compute_power_range[1]) * pow(10, 9)
        uav_transmission_rate = random.randint(uav_trans_rate_range[0], uav_trans_rate_range[1])
        uav_memory = random.randint(uav_memory_range[0], uav_memory_range[1])

        uavs_attr[kk, 0] = uav_computing_power
        uavs_attr[kk, 1] = uav_transmission_rate
        uavs_attr[kk, 2] = uav_memory
    df2 = pd.DataFrame(vehicles_attr)
    df2.to_csv('data/vehicles_attr.csv', header=False, index=False)

    df3 = pd.DataFrame(uavs_attr)
    df3.to_csv('data/uavs_attr.csv', header=False, index=False)


# 根据节点的编号，获取节点的属性值
# 车辆节点编号 0-99     雾节点编号 100-103   云端编号 104   无人机编号 105-110
def get_info_by_node_id(nid):
    # print(nid)
    test = nid
    if 0 <= nid <= 99:
        df = vehicles_attrs
        # 分别对应计算能力、传输率、节点内存
        info = [df.iloc[nid, 0], df.iloc[nid, 1], df.iloc[nid, 2]]
    elif 100 <= nid <= 103:
        df = fogs_attrs
        info = [df.iloc[nid - 100, 0], df.iloc[nid - 100, 1], df.iloc[nid - 100, 2]]
        # print(info)
    elif nid == 104:
        info = [cloud_compute_delay, cloud_transmission_rate, cloud_memory]
    elif 105 <= nid <= 110:
        df = uavs_attrs
        info = [df.iloc[nid - 105, 0], df.iloc[nid - 105, 1], df.iloc[nid - 105, 2]]
    # print(info)
    return info


# 根据染色体的实体数组，获取染色体的编码
# 每一个编码（基因）是子任务分配到的节点编号
def get_chromosome_link_code(obj_list):
    codes = []
    for e in obj_list:
        if isinstance(e, SubTask):
            codes.append(e.allocate_node_id)
    return codes


# 在给定染色体中，检查每一个节点的负载情况，返回双级列表
def get_each_node_load_situation(chromosomes, chromosome_id):
    one_link_obj = chromosomes[chromosome_id]
    nodes_receive_in_a_link = []
    for node_id in range(0, 105):
        special_node_receive = []
        for xx in range(one_link_obj.__len__()):
            if one_link_obj[xx].allocate_node_id == node_id:
                special_node_receive.append(one_link_obj[xx])
        nodes_receive_in_a_link.append(special_node_receive)
    return nodes_receive_in_a_link


# 给定一系列节点的承载状态，筛选出：任务过载的节点编号，过载量，导致节点过载的子任务(待修复的子任务)
def judge_nodes_overload(nodes_seq):
    overload_ids = []
    overload_values = []
    overload_sub_tasks = []
    for nid in range(nodes_seq.__len__()):
        node_loads = nodes_seq[nid]
        node_memory = get_info_by_node_id(nid)[2]
        cost_memory = 0
        for sub_e in node_loads:
            cost_memory += sub_e.storage_cost
            if cost_memory > node_memory:
                overload_ids.append(nid)
                overload_sub_tasks.append(sub_e)
        overload_values.append(cost_memory - node_memory)
        # print(nid, node_memory, cost_memory)
    # id 0:104（node）
    return overload_ids, overload_values, overload_sub_tasks


# 检查给定染色体中的过载情况，返回：染色体上过载节点集合，过载量， 待修复的子任务集合
def get_node_over_distribution(chromosomes, chromosome_id):
    nodes_receive_in_a_link = get_each_node_load_situation(chromosomes, chromosome_id)
    overload_nodes_set, nodes_over_values, prepare_repair_sub_objs = judge_nodes_overload(nodes_receive_in_a_link)
    return overload_nodes_set, nodes_over_values, prepare_repair_sub_objs


# 适应度函数，即计算染色体整体时延。
def compute_delay_in_a_link_v2(chro):
    num = 0
    parallel_execution = 0
    serial_execution = 0
    # print("计算时延：")
    # print(chro)
    chro = list(chro)
    ready_server_time = chro.pop()

    # print("进化过程ready_server_time" + str(ready_server_time))
    center = chro.pop()
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(chro)
    # 加上子任务发出时间到调度时间的差值 是为调度等待时间
    for ch in chro:
        ch.dispatch_wait_delay = ready_server_time - ch.timestamp

    # 获取所有的节点负载情况+求出了每个子任务的总时延
    nodes_loads = center.request_dicts
    # print(nodes_loads)
    # n_id代表卸载节点   load_dict子任务
    for n_id, load_dict in nodes_loads.items():
        if load_dict:  # 只考虑有负载的节点
            if n_id != cloud_id:
                n_compute_ability, n_transmission, _ = get_info_by_node_id(n_id)
                # print("load_dict:" + str(load_dict))
                upload_size = []
                load_numpy = []  # 表示染色体的第几个基因
                for kk, size in load_dict.items():
                    load_numpy.append(kk)
                    upload_size.append(chro[kk].upload_size)
                # print("old_load_numpy" + str(load_numpy))
                # print(upload_size)
                a = zip(upload_size, load_numpy)
                # print(a)
                upload_size, load_numpy = zip(*sorted(zip(upload_size, load_numpy)))
                # print("sorted_load_numpy" + str(load_numpy))
                # print(upload_size)
                for key in range(load_numpy.__len__()):
                    num += 1
                    # 原车辆的上传时延
                    # if chro[load_numpy[key]].belonging_vehicle_id == n_id:
                    #     chro[load_numpy[key]].up_delay = 0
                    # 无人机的上传时延
                    if chro[load_numpy[key]].belonging_uav_id == n_id:
                        chro[load_numpy[key]].up_delay = 0
                    else:
                        chro[load_numpy[key]].up_delay = upload_size[key] / (n_transmission / len(load_dict))
                    if key == 0:
                        chro[load_numpy[key]].wait_delay = 0
                        parallel_execution += 1
                        chro[load_numpy[key]].compute_delay = chro[load_numpy[key]].compute_cost / n_compute_ability
                        # print("n_compute_ability" + str(n_compute_ability))
                        # print("chro[load_numpy[key]].compute_cost" + str(chro[load_numpy[key]].compute_cost))
                        # print("计算时延：" + str(chro[load_numpy[key]].compute_delay))

                    else:
                        if chro[load_numpy[key - 1]].up_delay + chro[load_numpy[key - 1]].compute_delay + \
                                chro[load_numpy[key - 1]].wait_delay > chro[load_numpy[key]].up_delay:
                            chro[load_numpy[key]].wait_delay = chro[load_numpy[key - 1]].up_delay + chro[
                                load_numpy[key - 1]].compute_delay \
                                                               + chro[load_numpy[key - 1]].wait_delay - chro[
                                                                   load_numpy[key]].up_delay
                            serial_execution += 1
                            # parallel_execution -= 1 if parallel_execution >= 50 else 0
                            # print("计算等待时延:" + str(chro[load_numpy[key]].wait_delay))
                        else:
                            chro[load_numpy[key]].wait_delay = 0
                            parallel_execution += 1
                        chro[load_numpy[key]].compute_delay = chro[key].compute_cost / n_compute_ability
                        # print("n_compute_ability" + str(n_compute_ability))
                        # print("chro[load_numpy[key]].compute_cost" + str(chro[load_numpy[key]].compute_cost))
                        # print("计算时延：" + str(chro[load_numpy[key]].compute_delay))

                    is_reachable = calculate_two_vehicle_is_interact(
                        math.floor(
                            ready_server_time + chro[load_numpy[key]].up_delay + chro[load_numpy[key]].wait_delay +
                            chro[load_numpy[key]].compute_delay),
                        chro[load_numpy[key]].belonging_vehicle_id, chro[load_numpy[key]].allocate_node_id)
                    if not is_reachable:
                        chro[load_numpy[key]].punish_delay = 0.1
                    else:
                        chro[load_numpy[key]].punish_delay = 0

                    if chro[load_numpy[key]].belonging_vehicle_id == n_id:
                        chro[load_numpy[key]].down_delay = 0
                    else:
                        _, object_vehicle_transmission, _ = \
                            get_info_by_node_id(chro[load_numpy[key]].belonging_vehicle_id)
                        chro[load_numpy[key]].down_delay = upload_size[key] / object_vehicle_transmission

                    chro[load_numpy[key]].whole_delay = chro[load_numpy[key]].up_delay + chro[
                        load_numpy[key]].down_delay + \
                                                        chro[load_numpy[key]].wait_delay + \
                                                        chro[load_numpy[key]].compute_delay + \
                                                        chro[load_numpy[key]].punish_delay
            else:
                # cloud_compute_delay, cloud_transmission_rate, cloud_memory
                for key, storage in load_dict.items():
                    _, object_vehicle_transmission, _ = get_info_by_node_id(chro[key].belonging_vehicle_id)
                    chro[key].up_delay = chro[key].upload_size / (cloud_transmission_rate / len(load_dict))
                    chro[key].down_delay = chro[key].upload_size / object_vehicle_transmission
                    chro[key].wait_delay = 0
                    chro[key].compute_delay = 0
                    chro[key].punish_delay = 0
                    chro[key].whole_delay = chro[key].up_delay + chro[key].down_delay + \
                                            chro[key].wait_delay + \
                                            chro[key].compute_delay + \
                                            chro[key].punish_delay
    task_set = []
    for v_id in range(0, 100):
        sub_tasks_in_v = [chro[z] for z in range(chro.__len__()) if chro[z].belonging_vehicle_id == v_id]
        tasks_in_v = []
        for task_id in range(task_num_range[1] + 1):
            one_task_in_v = []
            for v_s_e in sub_tasks_in_v:
                if v_s_e.belonging_task_order == task_id:
                    one_task_in_v.append(v_s_e)
            if len(one_task_in_v):
                # print(len(one_task_in_v))
                tasks_in_v.append(one_task_in_v)
        task_set.append(tasks_in_v)

    a_link_delays = []
    a_link_comp_delays = []
    a_link_trans_delays = []
    a_link_wait_delays = []

    for z in task_set:
        for zz in z:
            all_subtask_whole_delay = [zz[x].whole_delay for x in range(zz.__len__())]
            max_value_local = all_subtask_whole_delay.index(max(all_subtask_whole_delay))
            comp_delay = zz[max_value_local].compute_delay
            trans_delay = zz[max_value_local].up_delay + zz[max_value_local].down_delay
            wait_delay = zz[max_value_local].wait_delay + zz[max_value_local].punish_delay
            single_task_delay = max([zz[x].whole_delay for x in range(zz.__len__())])
            a_link_comp_delays.append(comp_delay)
            a_link_trans_delays.append(trans_delay)
            a_link_wait_delays.append(wait_delay)
            a_link_delays.append(single_task_delay)
    task_average_comp_delay = sum(a_link_comp_delays) / len(a_link_comp_delays)
    task_average_trans_delay = sum(a_link_trans_delays) / len(a_link_trans_delays)
    task_average_wait_delay = sum(a_link_wait_delays) / len(a_link_wait_delays)
    task_average_delay = sum(a_link_delays) / len(a_link_delays)
    next_hop_delay = max(a_link_delays)

    chro.append(center)
    # print("num" + str(num))
    # print("parallel_execution" + str(parallel_execution))
    return chro, task_average_delay, next_hop_delay, parallel_execution, serial_execution, \
        task_average_comp_delay, task_average_trans_delay, task_average_wait_delay



def repair_overload_by_one_link_v2(link):
    delay_link, _, _, _, _, _, _, _ = compute_delay_in_a_link_v2(link)
    # print(delay_link)
    saved_center = delay_link[-1]
    # print(saved_center)
    total_up_delay = 0
    total_wait_delay = 0
    total_compute_delay = 0
    bad_genes = saved_center.self_detect_repair()
    for k in range(len(delay_link) - 1):
        total_up_delay += delay_link[k].up_delay
        total_wait_delay += delay_link[k].wait_delay
        total_compute_delay += delay_link[k].compute_delay
    average_up_delay = total_up_delay / (len(delay_link) - 1)
    average_compute_delay = total_compute_delay / (len(delay_link) - 1)
    average_wait_delay = total_wait_delay / (len(delay_link) - 1)
    if average_wait_delay + average_compute_delay < average_up_delay:
        for x in bad_genes:
            delay_link[x].allocate_node_id = saved_center.repair_malloc(x, link[x].storage_cost,
                                                                        link[x].selection_space)
    else:
        for e in bad_genes:
            delay_link[e].allocate_node_id = saved_center.repair_malloc1(e, link[e].storage_cost)
    # test
    assert len(saved_center.self_detect_repair()) == 0, '修复程序出现异常！'
    return list(delay_link)


# 修复工作，通过计算每个子任务的 通信和处理时延，来决定是在边缘端 还是云端卸载。
def repair_overload_by_one_link_v3(link):

    delay_link, _, _, _, _, _, _, _ = compute_delay_in_a_link_v2(link)
    # print(delay_link)
    saved_center = delay_link[-1]
    total_up_delay = 0
    total_wait_delay = 0
    total_compute_delay = 0
    bad_genes = saved_center.self_detect_repair()
    for k in range(len(delay_link) - 1):
        total_up_delay += delay_link[k].up_delay
        total_wait_delay += delay_link[k].wait_delay
        total_compute_delay += delay_link[k].compute_delay
    average_up_delay = total_up_delay / (len(delay_link) - 1)
    average_compute_delay = total_compute_delay / (len(delay_link) - 1)
    average_wait_delay = total_wait_delay / (len(delay_link) - 1)
    if average_wait_delay + average_compute_delay < average_up_delay:
        for x in bad_genes:
            delay_link[x].allocate_node_id = saved_center.repair_malloc(x, link[x].storage_cost,
                                                                        link[x].selection_space)
    else:
        for e in bad_genes:
            delay_link[e].allocate_node_id = saved_center.repair_malloc1(e, link[e].storage_cost)
    # test
    assert len(saved_center.self_detect_repair()) == 0, '修复程序出现异常！'
    return list(delay_link)


# def compute_delay_in_a_link_v2(chro):
#     num = 0
#     parallel_execution = 0
#     serial_execution = 0
#     # print("计算时延：")
#     # print(chro)
#     chro = list(chro)
#     ready_server_time = chro.pop()
#
#     # print("进化过程ready_server_time" + str(ready_server_time))
#     center = chro.pop()
#     # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     # print(chro)
#     # 加上子任务发出时间到调度时间的差值 是为调度等待时间
#     for ch in chro:
#         ch.dispatch_wait_delay = ready_server_time - ch.timestamp
#
#     # 获取所有的节点负载情况+求出了每个子任务的总时延
#     nodes_loads = center.request_dicts
#     # print(nodes_loads)
#     for n_id, load_dict in nodes_loads.items():
#         if load_dict:  # 只考虑有负载的节点
#             if n_id != cloud_id:
#                 n_compute_ability, n_tr, _ = get_info_by_node_id(n_id)
#                 # print("load_dict:" + str(load_dict))
#                 upload_size = []
#                 load_numpy = []  #表示染色体的第几个基因
#                 for kk, size in load_dict.items():
#                     load_numpy.append(kk)
#                     upload_size.append(chro[kk].upload_size)
#                 # print("old_load_numpy" + str(load_numpy))
#                 # print(upload_size)
#                 upload_size, load_numpy = zip(*sorted(zip(upload_size, load_numpy)))
#                 # print("sorted_load_numpy" + str(load_numpy))
#                 # print(upload_size)
#                 for key in range(load_numpy.__len__()):
#                     num += 1
#                     if chro[load_numpy[key]].belonging_vehicle_id == n_id:
#                         chro[load_numpy[key]].up_delay = 0
#                     else:
#                         chro[load_numpy[key]].up_delay = upload_size[key] / (n_tr / len(load_dict))
#
#                     if key == 0:
#                         chro[load_numpy[key]].wait_delay = 0
#                         parallel_execution += 1
#                         chro[load_numpy[key]].compute_delay = chro[load_numpy[key]].compute_cost / n_compute_ability
#                         # print("n_compute_ability" + str(n_compute_ability))
#                         # print("chro[load_numpy[key]].compute_cost" + str(chro[load_numpy[key]].compute_cost))
#                         # print("计算时延：" + str(chro[load_numpy[key]].compute_delay))
#
#                     else:
#                         if chro[load_numpy[key-1]].up_delay + chro[load_numpy[key-1]].compute_delay + \
#                                 chro[load_numpy[key-1]].wait_delay > chro[load_numpy[key]].up_delay:
#                             chro[load_numpy[key]].wait_delay = chro[load_numpy[key-1]].up_delay + chro[load_numpy[key-1]].compute_delay \
#                                                         + chro[load_numpy[key-1]].wait_delay - chro[load_numpy[key]].up_delay
#                             serial_execution += 1
#                             # print("计算等待时延:" + str(chro[load_numpy[key]].wait_delay))
#                         else:
#                             chro[load_numpy[key]].wait_delay = 0
#                             parallel_execution += 1
#                         chro[load_numpy[key]].compute_delay = chro[key].compute_cost / n_compute_ability
#                         # print("n_compute_ability" + str(n_compute_ability))
#                         # print("chro[load_numpy[key]].compute_cost" + str(chro[load_numpy[key]].compute_cost))
#                         # print("计算时延：" + str(chro[load_numpy[key]].compute_delay))
#
#                     is_reachable = calculate_two_vehicle_is_interact(
#                         math.floor(ready_server_time + chro[load_numpy[key]].up_delay + chro[load_numpy[key]].wait_delay +
#                                    chro[load_numpy[key]].compute_delay),
#                         chro[load_numpy[key]].belonging_vehicle_id, chro[load_numpy[key]].allocate_node_id)
#                     if not is_reachable:
#                         chro[load_numpy[key]].punish_delay = 0.3
#                     else:
#                         chro[load_numpy[key]].punish_delay = 0
#
#                     chro[load_numpy[key]].whole_delay = chro[load_numpy[key]].up_delay + chro[load_numpy[key]].wait_delay + \
#                                                  chro[load_numpy[key]].compute_delay + chro[load_numpy[key]].punish_delay + \
#                                                  chro[load_numpy[key]].dispatch_wait_delay
#             else:
#                 # cloud_compute_delay, cloud_transmission_rate, cloud_memory
#                 for key, storage in load_dict.items():
#                     chro[key].up_delay = chro[key].upload_size / (cloud_transmission_rate / len(load_dict))
#                     chro[key].wait_delay = 0
#                     chro[key].compute_delay = 0
#                     chro[key].punish_delay = 0
#                     chro[key].whole_delay = chro[key].up_delay + chro[key].wait_delay +\
#                                                  chro[key].compute_delay + chro[key].punish_delay + \
#                                                  chro[key].dispatch_wait_delay
#     task_set = []
#     for v_id in range(0, 100):
#         sub_tasks_in_v = [chro[z] for z in range(chro.__len__()) if chro[z].belonging_vehicle_id == v_id]
#         tasks_in_v = []
#         for task_id in range(task_num_range[1]):
#             one_task_in_v = []
#             for v_s_e in sub_tasks_in_v:
#                 if v_s_e.belonging_task_order == task_id:
#                     one_task_in_v.append(v_s_e)
#             if len(one_task_in_v):
#                 tasks_in_v.append(one_task_in_v)
#         task_set.append(tasks_in_v)
#
#     a_link_delays = []
#     # link_up_delays = [chro[k].up_delay for k in range(chro.__len__())]
#     # a_link_up_delays = sum(link_up_delays)/chro.__len__()
#     # link_wait_delays = [chro[k].wait_delay for k in range(chro.__len__())]
#     # a_link_wait_delays = sum(link_wait_delays)/chro.__len__()
#     # link_compute_delays = [chro[k].compute_delay for k in range(chro.__len__())]
#     # a_link_compute_delays = sum(link_compute_delays)/chro.__len__()
#     # link_punish_delays = [chro[k].punish_delay for k in range(chro.__len__())]
#     # a_link_punish_delays = sum(link_punish_delays)/chro.__len__()
#     # link_dispatch_wait_delay_delays = [chro[k].dispatch_wait_delay for k in range(chro.__len__())]
#     # a_link_dispatch_wait_delay_delays = sum(link_dispatch_wait_delay_delays)/chro.__len__()
#     # link_sub_task_delays = [chro[k].whole_delay for k in range(chro.__len__())]
#     # a_link_sub_task_delays = sum(link_sub_task_delays)/chro.__len__()
#     #
#     # print("a_link_up_delays:" + str(a_link_up_delays) + " " + "a_link_wait_delays" + str(a_link_wait_delays) + " " + "a_link_compute_delays" + str(a_link_compute_delays)
#     #       + " " + "a_link_punish_delays" + str(a_link_punish_delays) + " " + "a_link_dispatch_wait_delay_delays" + str(a_link_dispatch_wait_delay_delays) + " " + "a_link_sub_task_delays"
#     #       + str(a_link_sub_task_delays))
#
#     for z in task_set:
#         for zz in z:
#             single_task_delay = max([zz[x].whole_delay for x in range(zz.__len__())])
#             a_link_delays.append(single_task_delay)
#     task_average_delay = sum(a_link_delays) / len(a_link_delays)
#     next_hop_delay = max(a_link_delays)
#
#     chro.append(center)
#     # print("num" + str(num))
#     # print("parallel_execution" + str(parallel_execution))
#     return chro, task_average_delay, next_hop_delay, parallel_execution, serial_execution


# 在一条染色体上，每个车都随机选择2个子任务，按一定概率倒置编码
def reverse_2_point_by_vehicle_in_a_link(link):
    link_center = link[-1]

    for v_id in range(0, 100):
        sub_tasks_in_v = [link[z] for z in range(link.__len__() - 1) if link[z].belonging_vehicle_id == v_id]

        if len(sub_tasks_in_v) < 2:
            continue

        if len(sub_tasks_in_v):
            is_reverse = random.random()
            if is_reverse > p_reverse:
                sample_range = range(0, int(sub_tasks_in_v.__len__()))
                s1, s2 = random.sample(sample_range, 2)
                index_1 = min(s1, s2)
                index_2 = max(s1, s2)

                old_nodes_seq = copy.deepcopy([sub_tasks_in_v[x].allocate_node_id for x in range(index_1, index_2 + 1)])
                old_nodes_seq.reverse()
                for x_v in range(index_1, index_2 + 1):
                    link_center.free_and_assign_malloc(request_id=sub_tasks_in_v[x_v].id_in_link,
                                                       request_size=sub_tasks_in_v[x_v].storage_cost,
                                                       old_node_id=sub_tasks_in_v[x_v].allocate_node_id,
                                                       assign_node_id=old_nodes_seq[x_v - index_1])
                    sub_tasks_in_v[x_v].allocate_node_id = old_nodes_seq[x_v - index_1]


# 按一定概率进行变异
def link_mutation_after_cross(link):
    test_threshold = 0.01 * (link.__len__())
    link_center = link[-1]
    for gene_id in range(len(link) - 1):
        is_mutation = random.random()
        if is_mutation <= p_mutation:
            now_allocated = link[gene_id].allocate_node_id
            while True:
                new_allocate = random.choice(link[gene_id].selection_space)
                if new_allocate == 104 and test_threshold > 0:
                    test_threshold -= 1
                    continue
                else:
                    if new_allocate != now_allocated:
                        link_center.free_and_assign_malloc(request_id=gene_id, request_size=link[gene_id].storage_cost,
                                                           old_node_id=now_allocated, assign_node_id=new_allocate)
                        link[gene_id].allocate_node_id = new_allocate
                        break


# # 按条件选择2条较好的染色体(时延比较低的)，进行交叉、变异和修复生成
# def sample_and_cross_covariance_generating(chromosomes, chromosomes_delays):
#
#     if a_test_chros_normal(chromosomes) is False:
#         print('error---------------------------------------------------!')
#
#     r_list = []
#     k_smallest = 2
#
#     # 锦标赛算法，先从所有染色体中选择10条，从10条里选时延最小的两条
#     for x in range(10):
#         r_list.append(random.randint(0, 99))
#     selected_chromosomes_delays = [chromosomes_delays[ad_i] for ad_i in r_list]
#     indices = np.argpartition(selected_chromosomes_delays, k_smallest)[:k_smallest]
#
#     chromosome_1 = copy.deepcopy(chromosomes[indices[0]])
#     chromosome_2 = copy.deepcopy(chromosomes[indices[1]])
#     # print(chromosome_1)
#     center_1 = chromosome_1[-1]
#     center_2 = chromosome_2[-1]
#
#     chromosome1_node0 = []
#     chromosome2_node0 = []
#     for l in range(chromosome_1.__len__() - 1):
#         chromosome1_node0.append(chromosome_1[l].allocate_node_id)
#         chromosome2_node0.append(chromosome_2[l].allocate_node_id)
#     print("chromosome1_node0" + str(chromosome1_node0))
#     print("chromosome2_node0" + str(chromosome2_node0))
#
#     for v_id in range(0, 100):
#         # gene_order = [z for z in range(chromosome_1.__len__() - 1) if chromosome_1[z].belonging_vehicle_id == v_id]
#         sub_tasks_in_v1 = [chromosome_1[z] for z in range(chromosome_1.__len__() - 1) if chromosome_1[z].belonging_vehicle_id == v_id]
#         sub_tasks_in_v2 = [chromosome_2[x] for x in range(chromosome_2.__len__() - 1) if chromosome_1[x].belonging_vehicle_id == v_id]
#         if(len(sub_tasks_in_v1)):
#             n_cross = random.random()
#             cut_a = random.randint(0, len(sub_tasks_in_v1))
#             cut_b = random.randint(0, len(sub_tasks_in_v2))
#             for gene_id in range(min(cut_a, cut_b), max(cut_a, cut_b)):
#                 if n_cross <= p_cross:
#                     gene1_node = chromosome_1[gene_id].allocate_node_id
#                     gene2_node = chromosome_2[gene_id].allocate_node_id
#                     gene1_size = chromosome_1[gene_id].storage_cost
#                     gene2_size = chromosome_2[gene_id].storage_cost
#                     center_1.free_and_assign_malloc(request_id=gene_id, request_size=gene1_size, old_node_id=gene1_node,
#                                             assign_node_id=gene2_node)
#                     chromosome_1[gene_id].allocate_node_id = gene2_node
#
#                     center_2.free_and_assign_malloc(request_id=gene_id, request_size=gene2_size, old_node_id=gene2_node,
#                                             assign_node_id=gene1_node)
#                     chromosome_2[gene_id].allocate_node_id = gene1_node
#     chromosome1_node = []
#     chromosome2_node = []
#     for l in range(chromosome_1.__len__() - 1):
#         chromosome1_node.append(chromosome_1[l].allocate_node_id)
#         chromosome2_node.append(chromosome_2[l].allocate_node_id)
#     print("chromosome1_node" + str(chromosome1_node))
#     print("chromosome2_node" + str(chromosome2_node))
#
#
#     reverse_2_point_by_vehicle_in_a_link(chromosome_1)
#     reverse_2_point_by_vehicle_in_a_link(chromosome_2)
#
#     link_mutation_after_cross(chromosome_1)
#     link_mutation_after_cross(chromosome_2)
#
#     return chromosome_1, chromosome_2

# 按条件选择2条较好的染色体(时延比较低的)，进行交叉、变异和修复生成
def sample_and_cross_covariance_generating(chromosomes, chromosomes_delays):
    if a_test_chros_normal(chromosomes) is False:
        print('error---------------------------------------------------!')

    r_list = []
    k_smallest = 2

    # 锦标赛算法，先从所有染色体中选择10条，从10条里选时延最小的两条
    for x in range(10):
        r_list.append(random.randint(0, 99))
    selected_chromosomes_delays = [chromosomes_delays[ad_i] for ad_i in r_list]
    indices = np.argpartition(selected_chromosomes_delays, k_smallest)[:k_smallest]

    chromosome_1 = copy.deepcopy(chromosomes[indices[0]])
    chromosome_2 = copy.deepcopy(chromosomes[indices[1]])
    # print(chromosome_1)
    center_1 = chromosome_1[-1]
    center_2 = chromosome_2[-1]

    for v_id in range(0, 100):
        gene_order = [z for z in range(chromosome_1.__len__() - 1) if
                      chromosome_1[z].belonging_vehicle_id == v_id]
        sub_tasks_in_v1 = [chromosome_1[z] for z in range(chromosome_1.__len__() - 1) if
                           chromosome_1[z].belonging_vehicle_id == v_id]
        sub_tasks_in_v2 = [chromosome_2[x] for x in range(chromosome_2.__len__() - 1) if
                           chromosome_1[x].belonging_vehicle_id == v_id]
        if (len(sub_tasks_in_v1)):
            n_cross = random.random()
            cut_a = random.randint(0, len(sub_tasks_in_v1))
            cut_b = random.randint(0, len(sub_tasks_in_v2))
            for gene_id in range(min(cut_a, cut_b), max(cut_a, cut_b)):
                if n_cross <= p_cross:
                    gene1_node = chromosome_1[gene_order[gene_id]].allocate_node_id
                    gene2_node = chromosome_2[gene_order[gene_id]].allocate_node_id
                    gene1_size = chromosome_1[gene_order[gene_id]].storage_cost
                    gene2_size = chromosome_2[gene_order[gene_id]].storage_cost
                    center_1.free_and_assign_malloc(request_id=gene_order[gene_id], request_size=gene1_size,
                                                    old_node_id=gene1_node,
                                                    assign_node_id=gene2_node)
                    chromosome_1[gene_order[gene_id]].allocate_node_id = gene2_node

                    center_2.free_and_assign_malloc(request_id=gene_order[gene_id], request_size=gene2_size,
                                                    old_node_id=gene2_node,
                                                    assign_node_id=gene1_node)
                    chromosome_2[gene_order[gene_id]].allocate_node_id = gene1_node
    # chromosome1_node = []
    # chromosome2_node = []
    # for l in range(chromosome_1.__len__() - 1):
    #     chromosome1_node.append(chromosome_1[l].allocate_node_id)
    #     chromosome2_node.append(chromosome_2[l].allocate_node_id)
    # print("chromosome1_node" + str(chromosome1_node))
    # print("chromosome2_node" + str(chromosome2_node))

    reverse_2_point_by_vehicle_in_a_link(chromosome_1)
    reverse_2_point_by_vehicle_in_a_link(chromosome_2)

    link_mutation_after_cross(chromosome_1)
    link_mutation_after_cross(chromosome_2)

    return chromosome_1, chromosome_2


def init_100_chromosomes(all_genes, init_time):
    print('初始化100条染色体...')
    choices_100 = []
    source_center = SourceCenter(origin_source)
    for cn in range(100):
        chromosome_link_obj = []
        # all_genes 确定的子任务数量
        test_threshold = 0.01 * (len(all_genes))
        # print("test_threshold", test_threshold)
        for gene_idx in range(all_genes.__len__()):
            # print("all_genes.__len__()", all_genes.__len__())
            all_genes[gene_idx].id_in_link = gene_idx
            # 得到当前这个子任务分配的节点
            # print("selection_space", all_genes[gene_idx].selection_space)
            all_genes[gene_idx].allocate_node_id = \
                source_center.init_malloc(gene_idx, all_genes[gene_idx].storage_cost,
                                          all_genes[gene_idx].selection_space, test_threshold)
            chromosome_link_obj.append(copy.deepcopy(all_genes[gene_idx]))

        # 注意：染色体的倒数第2个对象是资源分配中心,最后一个对象是调度时间
        # print(source_center.request_dicts[18])
        chromosome_link_obj.append(copy.deepcopy(source_center))
        chromosome_link_obj.append(init_time)
        # print(chromosome_link_obj[-2].request_dicts[18])
        choices_100.append(chromosome_link_obj)
        source_center.reset()

    del source_center
    del all_genes
    return choices_100


def initialization(dispatch_pre_time, dispatch_time):
    # 时间 间隔国定

    # 某个时刻，某辆车的不合理子任务数量， 不合理是考虑在当前时刻是否还在范围内。
    unsure_subtasks = []
    # 某个时刻，某辆车的合理子任务数量。
    sure_subtasks = []
    sure_vehicles_id = []
    unsure_tasks = []
    sure_tasks = []
    # 先检测调度之前的任务请求情况
    for tx in range(dispatch_pre_time, dispatch_time + 1):
        tx_total_vehicle_num_in_area = data.loc[tx].shape[0]  # tx 时间下区域内的总车辆数目
        tx_all_vehicles_sub_objects = []  # 在tx时间下，保存所有车辆发出的子任务
        tx_all_vehicles_task_objects = []  # 在tx时间下，保存所有车辆发出的任务
        tx_task_num = 0
        tx_vehicles_id = []
        for i in range(tx_total_vehicle_num_in_area):
            # 当前车辆的信息
            current_vehicle_id = data.loc[tx].iloc[i, 0]
            tx_vehicles_id.append(current_vehicle_id)
            # 根据车辆ID 获取相应信息。
            try:
                fi = 'request_data/time' + str(tx) + '-vehicle' + str(current_vehicle_id) + 'request.data'
                with open(fi, 'rb') as f:
                    i_task_group = pickle.load(f)
                    # print("i_task_group info", tx, i_task_group.vehicle_id, i_task_group.task_num)
                    tx_task_num += i_task_group.task_num

                if i_task_group.task_num > 0:
                    tx_all_vehicles_task_objects.extend(i_task_group.task_pool)
                    tx_all_vehicles_sub_objects.extend(i_task_group.get_sub_task_objects())
            except FileNotFoundError:
                continue
        # 前面任意tx时刻，可能存在离开地图的车辆，任务就可能丢弃
        if tx != dispatch_time:
            unsure_tasks.extend(tx_all_vehicles_task_objects)
            unsure_subtasks.extend(tx_all_vehicles_sub_objects)
        # 最后时刻，不可能存在离开地图的车辆，任务就不可能丢弃
        else:
            sure_tasks.extend(tx_all_vehicles_task_objects)
            sure_subtasks.extend(tx_all_vehicles_sub_objects)
            sure_vehicles_id = tx_vehicles_id
    # print("当前车辆请求的子任务： " + str(sure_subtasks))
    # print(unsure_task_num, sure_task_num)
    for ff in unsure_tasks:
        if ff.belonging_vehicle_id in sure_vehicles_id:
            sure_tasks.append(ff)
            unsure_tasks.remove(ff)
    # print("需要被服务的子任务： " + str(sure_subtasks))
    global total_request_task_num, severed_task_num
    total_request_task_num += len(unsure_tasks) + len(sure_tasks)
    severed_task_num += len(sure_tasks)
    print("总共的任务数和被服务的任务数为：" + str(total_request_task_num) + " " + str(severed_task_num))
    vs = data_uav.loc[dispatch_time]
    vs = vs.set_index('uav_id')
    for e in unsure_subtasks:
        if e.belonging_vehicle_id in sure_vehicles_id:
            ux = vs.loc[e.belonging_uav_id][0]
            uy = vs.loc[e.belonging_uav_id][1]
            # 无人机的通信空间
            e.selection_space = compute_distance_and_save_usable_nodes(ux, uy, dispatch_time)

            # 过去的请求将在dispatch时刻被调度，需要更新其通讯空间
            # vs = data.loc[dispatch_time]
            # vs = vs.set_index('vehicle_id')
            # dx = vs.loc[e.belonging_vehicle_id][0]
            # dy = vs.loc[e.belonging_vehicle_id][1]
            # ux, uy, uav_id = search_auxiliary_uav_location(dx, dy, dispatch_time)
            # 原车辆的通信空间
            # e.selection_space = compute_distance_and_save_usable_nodes(dx, dy, dispatch_time)

            sure_subtasks.append(e)
            unsure_subtasks.remove(e)

    print('时间' + str(dispatch_pre_time) + '~' + str(dispatch_time) + '内产生的总子任务数目为：' +
          str(len(unsure_subtasks) + len(sure_subtasks)) + ', 其中在' + str(
        dispatch_time) + '时刻可调度的子任务数目为' +
          str(len(sure_subtasks)) + ',不可调度子任务数为：' + str(len(unsure_subtasks)))

    print('时间' + str(dispatch_time) + ' 下的基因数：' + str(sure_subtasks.__len__()))

    if len(sure_subtasks) != 0:
        # 根据当前时间段的初始状态生成100条染色体(随机分配，需要修复)
        initial_seeds = init_100_chromosomes(sure_subtasks, dispatch_time)

        if a_test_chros_normal(initial_seeds) is False:
            print('初始化的染色体有错！！！！！！！！！！！！！！！！！！')

        repaired_initial_seeds = []
        # 修复100条初代染色体，并计算时延
        with futures.ProcessPoolExecutor(5) as pool:
            # 修复了随机生成的100条染色体的过大负载问题。
            generator1 = pool.map(repair_overload_by_one_link_v2, (tuple(x) for x in initial_seeds))
            for g in generator1:
                repaired_initial_seeds.append(g)

        if a_test_chros_normal(repaired_initial_seeds) is False:
            print('修复以后的染色体有错！！！！！！！！！！！！！！！！！！')

        initial_genetic_delays = []
        initial_hop_delays = []
        serial_execution = []
        parallel_execution = []
        initial_genetic_comp_delays = []
        initial_genetic_trans_delays = []
        initial_genetic_wait_delays = []

        delayed_repaired_initial_seeds = []
        for k in repaired_initial_seeds:
            k.append(dispatch_time)
        # print("====")
        # print(repaired_initial_seeds)
        with futures.ProcessPoolExecutor(5) as pool0:
            generator = pool0.map(compute_delay_in_a_link_v2, (tuple(x) for x in repaired_initial_seeds))
            for x in generator:
                delayed_repaired_initial_seeds.append(x[0])
                initial_genetic_delays.append(x[1])
                initial_hop_delays.append(x[2])
                parallel_execution.append(x[3])
                serial_execution.append(x[4])
                initial_genetic_comp_delays.append(x[5])
                initial_genetic_trans_delays.append(x[6])
                initial_genetic_wait_delays.append(x[7])

        if a_test_chros_normal(delayed_repaired_initial_seeds) is False:
            print('修复+计算以后的染色体有错！！！！！！！！！！！！！！！！！！')

    else:
        delayed_repaired_initial_seeds = []
        initial_genetic_delays = [0]
        initial_hop_delays = [1]
        initial_genetic_comp_delays = [0]
        initial_genetic_trans_delays = [0]
        initial_genetic_wait_delays = [0]

    # for mm in range(delayed_repaired_initial_seeds[1].__len__() - 2):
    #     print("修复后" + str(delayed_repaired_initial_seeds[1][mm].allocate_node_id))
    # print("=========================")

    return delayed_repaired_initial_seeds, initial_genetic_delays, initial_hop_delays, parallel_execution, \
        serial_execution, initial_genetic_comp_delays, initial_genetic_trans_delays, initial_genetic_wait_delays


# 算法开始进化
def evolution(time_step, origin_chromosomes, origin_genetic_delays, origin_hop_delays, origin_parallel_execution,
              origin_serial_execution, origin_genetic_comp_delays, origin_genetic_trans_delays,
              origin_genetic_wait_delays):
    if a_test_chros_normal(origin_chromosomes) is False:
        print('演化开始的染色体有错！！！！！！！！！！！！！！！！！！')

    ev_start_t = time.time()
    next_chromosomes = []
    top_100_chromosomes = []
    # 产生50条新的染色体
    cons_s_t = time.time()
    for make_epoch in range(25):
        gen_1, gen_2 = sample_and_cross_covariance_generating(origin_chromosomes, origin_genetic_delays)
        next_chromosomes.append(gen_1)
        next_chromosomes.append(gen_2)
    cons_e_t = time.time()
    print('构造50条完成耗时：' + str(cons_e_t - cons_s_t))

    if a_test_chros_normal(next_chromosomes) is False:
        print('演化构造50的染色体有错！！！！！！！！！！！！！！！！！！')

    repaired_next_chromosomes = []
    re_start_t = time.time()
    for k in next_chromosomes:
        k.append(time_step)
    with futures.ProcessPoolExecutor(5) as pool:
        generator2 = pool.map(repair_overload_by_one_link_v3, (tuple(e) for e in next_chromosomes))
        for g2 in generator2:
            repaired_next_chromosomes.append(g2)

    if a_test_chros_normal(repaired_next_chromosomes) is False:
        print('演化修复50的染色体有错！！！！！！！！！！！！！！！！！！')

    re_end_t = time.time()
    repair_time = re_end_t - re_start_t
    print("修复时间为 " + str(repair_time))
    next_genetic_delays = []
    next_hop_delays = []
    next_genetic_comp_delays = []
    next_genetic_trans_delays = []
    next_genetic_wait_delays = []

    delay_st_time = time.time()

    for e in repaired_next_chromosomes:
        e.append(time_step)
    next_parallel_execution = []
    next_serial_execution = []
    delayed_repaired_next_chromosomes = []
    with futures.ProcessPoolExecutor(5) as pool1:
        generator = pool1.map(compute_delay_in_a_link_v2, repaired_next_chromosomes)
        for x in generator:
            delayed_repaired_next_chromosomes.append(x[0])
            next_genetic_delays.append(x[1])
            next_hop_delays.append(x[2])
            next_parallel_execution.append(x[3])
            next_serial_execution.append(x[4])
            next_genetic_comp_delays.append(x[5])
            next_genetic_trans_delays.append(x[6])
            next_genetic_wait_delays.append(x[7])

    # print("next_genetic_delays")
    # print(next_genetic_delays)
    if a_test_chros_normal(delayed_repaired_next_chromosomes) is False:
        print('演化计算50的染色体有错！！！！！！！！！！！！！！！！！！')

    delay_ed_time = time.time()
    delay_time = delay_ed_time - delay_st_time
    print("计算时延的时长为 " + str(delay_time))

    merged_genetic_delays = origin_genetic_delays + next_genetic_delays
    merged_genetic_comp_delays = origin_genetic_comp_delays + next_genetic_comp_delays
    merged_genetic_trans_delays = origin_genetic_trans_delays + next_genetic_trans_delays
    merged_genetic_wait_delays = origin_genetic_wait_delays + next_genetic_wait_delays

    # print("merged_genetic_delays" + str(merged_genetic_delays))
    merged_hop_delays = origin_hop_delays + next_hop_delays
    parallel_execution_radio = origin_parallel_execution + next_parallel_execution
    serial_execution_radio = origin_serial_execution + next_serial_execution

    # 选用100个最低延时的染色体
    select_st_time = time.time()
    top_100_indices = np.argpartition(merged_genetic_delays, 100)[:100]

    for n_s_idx in top_100_indices:
        if 0 <= n_s_idx <= 99:
            top_100_chromosomes.append(copy.deepcopy(origin_chromosomes[n_s_idx]))
        elif 100 <= n_s_idx <= 149:
            top_100_chromosomes.append(copy.deepcopy(delayed_repaired_next_chromosomes[n_s_idx - 100]))

    select_ed_time = time.time()
    select_time = select_ed_time - select_st_time
    print("选择100条染色体的时长为 " + str(select_time))

    top_100_genetic_delays = [merged_genetic_delays[sd] for sd in top_100_indices]
    top_100_genetic_comp_delays = [merged_genetic_comp_delays[c] for c in top_100_indices]
    top_100_genetic_trans_delays = [merged_genetic_trans_delays[t] for t in top_100_indices]
    top_100_genetic_wait_delays = [merged_genetic_wait_delays[w] for w in top_100_indices]
    top_100_hop_delays = [merged_hop_delays[shd] for shd in top_100_indices]
    top_100_parallel_execution = [parallel_execution_radio[lhd] for lhd in top_100_indices]
    top_100_serial_execution = [serial_execution_radio[mhd] for mhd in top_100_indices]

    ev_end_t = time.time()
    print('单次演化时间：' + str(ev_end_t - ev_start_t))

    # 不清楚什么含义。
    for ch in range(top_100_chromosomes.__len__()):
        allocate_id = []
        for ee in range(top_100_chromosomes[ch].__len__() - 1):
            allocate_id.append(top_100_chromosomes[ch][ee].allocate_node_id)
    return top_100_chromosomes, top_100_genetic_delays, top_100_hop_delays, top_100_parallel_execution,\
        top_100_serial_execution, top_100_genetic_comp_delays, top_100_genetic_trans_delays, top_100_genetic_wait_delays


def save_excellent_gene_list(time_s, selected_chromosome):
    f = os.getcwd() + "//result"
    if not os.path.exists(f):
        os.mkdir(f)
    fo = 'result/time' + str(time_s) + '-best_chromosome.data'
    with open(fo, 'wb+') as f:
        pickle.dump(obj=selected_chromosome, file=f)


# 验证染色体的合法性
def a_test_chros_normal(chros):
    result = True
    dt = -10000
    for e in chros:
        if isinstance(e[-1], int):
            dt = e.pop()
        sc = e.pop()
        for g in e:
            n_id = g.allocate_node_id
            g_id = g.id_in_link
            if g_id not in sc.request_dicts[n_id].keys():
                result = False
        e.append(sc)
        if dt > 0:
            e.append(dt)
    return result


def severd_ratio():
    global total_request_task_num, severed_task_num
    severd_ratio = severed_task_num / total_request_task_num
    return severd_ratio
