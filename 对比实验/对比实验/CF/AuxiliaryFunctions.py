"""
@description: 辅助函数
"""
import numpy as np
import random
import copy
import math
import pickle
import pandas as pd
import math
from EntityDefinition import TaskGroup, Task
from Configuration import fog_compute_power_range, fog_trans_rate_range, fog_memory_range, v_compute_power_range, \
    v_trans_rate_range, v_memory_range, cloud_compute_delay, cloud_transmission_rate, cloud_memory, cloud_id, \
    cloud_accept_threshold, task_num_range, p_cross, p_mutation, p_reverse, data, v2f_limit_distance, \
    v2v_limit_distance, fogs_location,  sub_task_num_range, upload_size_range, lam, \
    download_size_range, compute_cost_range, storage_cost_range, total_tasks_number, tasks_pool_path, tasks_table_path, vehicles_attr


import sys

sys.setrecursionlimit(10000)
total_request_task_num = 0
severed_task_num = 0
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


# def construct_tasks_table_and_update_tasks_pool():
#     tasks_table = np.zeros((total_tasks_number, sub_task_num_range[1] * 4 + 2))
#     tasks_table[:, :] = None
#     print(tasks_table.shape)
#     tasks_pool = []
#     for i in range(total_tasks_number):
#
#         i_identifier = i
#         i_sub_tasks_num = random.randint(sub_task_num_range[0], sub_task_num_range[1])
#         i_task_info_dict = dict()
#
#         tasks_table[i, 0] = i_identifier
#         tasks_table[i, 1] = i_sub_tasks_num
#
#         for j in range(i_sub_tasks_num):
#             us = 0.01 * random.randint(int(upload_size_range[0]*100), int(upload_size_range[1]*100))
#             ds = random.randint(download_size_range[0], download_size_range[1])
#             cc = random.randint(compute_cost_range[0], compute_cost_range[1]) * pow(10, 8)
#             sc = random.randint(storage_cost_range[0], storage_cost_range[1])
#             tasks_table[i, 2+j*4:2+(j+1)*4] = [us, ds, cc, sc]
#             i_task_info_dict['sub_task_'+str(j)] = [us, ds, cc, sc]
#         create_task = Task(timestamp=-404, vehicle_id=-404, task_order=-404, communication_space=[],
#                            sub_tasks_num=i_sub_tasks_num, task_info_dict=i_task_info_dict)
#         tasks_pool.append(create_task)
#
#     sub_titles = ['task_identifier', 'sub_tasks_num']
#     for x in range(sub_task_num_range[1]):
#         sub_titles.append('sub_task_'+str(x)+'.up_size')
#         sub_titles.append('sub_task_'+str(x)+'.down_size')
#         sub_titles.append('sub_task_' + str(x) + '.compute_cost')
#         sub_titles.append('sub_task_' + str(x) + '.storage_cost')
#
#     tasks_table_df = pd.DataFrame(tasks_table, index=None, columns=sub_titles)
#
#     tasks_table_df.to_csv(tasks_table_path, index=False)
#     fo = tasks_pool_path
#     with open(fo, 'wb+') as f:
#         pickle.dump(obj=tasks_pool, file=f)
#
#
# # 预先抽取所有时间下的任务
# def tasks_release():
#
#     f1 = tasks_pool_path
#     with open(f1, 'rb') as f:
#         tasks_pool = pickle.load(f)
#
#     repeat_filter = np.zeros((100, 300)) - 1
#     vehicle_request_times = []
#     for vvv in range(0, 100):
#         vehicle_i_request_time = []
#         interval_total_time = 1
#         while interval_total_time < 301:
#             vehicle_interval = int(np.random.exponential(lam) + 1)
#             vehicle_i_request_time.append(interval_total_time)
#             interval_total_time += vehicle_interval
#         vehicle_request_times.append(vehicle_i_request_time)
#
#     for t in range(1, 301):
#         total_vehicle_num_in_area = data.loc[t].shape[0]
#
#         for v in range(total_vehicle_num_in_area):
#
#             vehicle_id = data.loc[t].iloc[v, 0]
#             vehicle_x = data.loc[t].iloc[v, 1]
#             vehicle_y = data.loc[t].iloc[v, 2]
#             # vehicle_interval = int(vehicles_attrs.iloc[vehicle_id-1, 3])
#
#             # print(vehicle_request_times)
#                 # vehicle_request_times = list(range(1, 301, vehicle_interval))
#
#             if t in vehicle_request_times[v]:
#                 communication_space = compute_distance_and_save_usable_nodes(v_x=vehicle_x, v_y=vehicle_y, time_step=t)
#                 t_v_task_num = random.randint(task_num_range[0], task_num_range[1])
#                 task_set = []
#                 count = 0
#                 for i in range(t_v_task_num):
#                     while True:
#                         i_random_idx = random.randint(0, total_tasks_number - 1)
#                         i_select = tasks_pool[i_random_idx]
#                         if i_random_idx not in repeat_filter[vehicle_id - 1, 0:t - 1]:
#                             repeat_filter[vehicle_id - 1, t - 1] = i_random_idx
#                             count += 1
#
#                             i_ready = copy.deepcopy(i_select)
#                             i_ready.update_task_part_attr(timestamp=t, vehicle_id=vehicle_id, task_order=count,
#                                                           communication_space=copy.deepcopy(communication_space))
#
#                             task_set.append(i_ready)
#                             break
#
#                 tg = TaskGroup(timestamp=t, vehicle_id=vehicle_id, task_pool=task_set,
#                                communication_space=copy.deepcopy(communication_space))
#                 print(tg.get_all_information())
#                 fv = 'request_data/time' + str(t) + '-vehicle' + str(vehicle_id) + 'request.data'
#                 print(fv)
#                 with open(fv, 'wb+') as f:
#                     pickle.dump(obj=tg, file=f)


# # 为车辆节点和雾节点生成属性：计算能力，传输速率， 存储空间
# def generate_attribution_for_each_node():
#     fogs_attr = np.zeros((4, 3))
#     for ii in range(0, 4):
#         fog_computing_power = random.randint(fog_compute_power_range[0], fog_compute_power_range[1]) * pow(10, 8)
#         fog_transmission_rate = random.randint(fog_trans_rate_range[0], fog_trans_rate_range[1])
#         fog_memory = random.randint(fog_memory_range[0], fog_memory_range[1])
#
#         fogs_attr[ii, 0] = fog_computing_power
#         fogs_attr[ii, 1] = fog_transmission_rate
#         fogs_attr[ii, 2] = fog_memory
#
#     df1 = pd.DataFrame(fogs_attr)
#     df1.to_csv('data/fogs_attr.csv', header=False, index=False)
#
#     vehicles_attr = np.zeros((100, 4))
#     for jj in range(0, 100):
#         vehicle_computing_power = random.randint(v_compute_power_range[0], v_compute_power_range[1]) * pow(10, 8)
#         vehicle_transmission_rate = random.randint(v_trans_rate_range[0], v_trans_rate_range[1])
#         vehicle_memory = random.randint(v_memory_range[0], v_memory_range[1])
#         # vehicle_interval = random.randint(v_request_interval_range[0], v_request_interval_range[1])
#
#         vehicles_attr[jj, 0] = vehicle_computing_power
#         vehicles_attr[jj, 1] = vehicle_transmission_rate
#         vehicles_attr[jj, 2] = vehicle_memory
#         # vehicles_attr[jj, 3] = vehicle_interval
#
#     df2 = pd.DataFrame(vehicles_attr)
#     df2.to_csv('data/vehicles_attr.csv', header=False, index=False)


# 根据节点的编号，获取节点的属性值
# 车辆节点编号 0-99     雾节点编号 100-103   云端编号 104
def get_info_by_node_id(nid):
    # print(nid)
    if 0 <= nid <= 99:
        df = pd.read_csv('data/vehicles_attr.csv', header=None, index_col=None)
        # 分别对应计算能力、传输率、节点内存
        info = [df.iloc[nid, 0], df.iloc[nid, 1], df.iloc[nid, 2]]
    elif 100 <= nid <= 103:
        df = pd.read_csv('data/fogs_attr.csv', header=None, index_col=None)
        info = [df.iloc[nid - 100, 0], df.iloc[nid - 100, 1], df.iloc[nid - 100, 2]]
        # print(info)
    elif nid == 104:
        info = [cloud_compute_delay, cloud_transmission_rate, cloud_memory]
    # print(info)
    return info


def get_node_memory():
    # print(nid)
    mem_array = np.zeros(105, dtype=int)
    # print(mem_array)
    for nid in range(0, 105):
        if 0 <= nid <= 99:
            df = pd.read_csv('data/vehicles_attr.csv', header=None, index_col=None)
            mem_array[nid] = df.iloc[nid, 2]

        elif 100 <= nid <= 103:
            df = pd.read_csv('data/fogs_attr.csv', header=None, index_col=None)
            mem_array[nid] = df.iloc[nid-100, 2]
        elif nid == 104:
            mem_array[nid] = cloud_memory

    return mem_array


# 根据染色体的实体数组，获取染色体的编码
# 每一个编码（基因）是子任务分配到的节点编号
def get_chromosome_link_code(obj_list):
    codes = []
    for e in obj_list:
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


# # 修复某一条染色体
# def repair_overload_by_one_link(links, link_id, bad_sub_tasks):
#     # re-allocate
#     for e in bad_sub_tasks:
#         print('修复执行？')
#         current_selection = e.allocate_node_id
#         allow_set = e.selection_space
#         cloud_request_count = 0
#         while True:
#             new_selection = random.choice(allow_set)
#             if new_selection == cloud_id:
#                 cloud_request_count += 1
#
#             if new_selection != current_selection:
#                 if new_selection != cloud_id:
#                     e.allocate_node_id = new_selection
#                     break
#                 elif new_selection == cloud_id and cloud_request_count > cloud_accept_threshold:
#                     e.allocate_node_id = new_selection
#                     break
#     # test
#     _, _, new_bad_set = get_node_over_distribution(chromosomes=links, chromosome_id=link_id)
#     if len(new_bad_set):
#         repair_overload_by_one_link(links, link_id, new_bad_set)


# 计算一条染色体方案对应的时延
# def compute_delay_in_a_link(redeay_server_time, chromosomes, chromosome_id):
#
#     # 加上子任务发出时间到调度时间的差值 是为调度等待时间
#     for ch in chromosomes[chromosome_id]:
#         ch.dispatch_wait_delay = redeay_server_time - ch.timestamp
#
#     # 获取所有的节点负载情况
#     nodes_loads = get_each_node_load_situation(chromosomes, chromosome_id)
#     for n_id in range(nodes_loads.__len__()):
#         if len(nodes_loads[n_id]):  # 只考虑有负载的节点
#             if n_id != 104:
#                 # 非云节点，获取计算能力与传输速率
#                 n_compute_ability, n_tr, _ = get_info_by_node_id(n_id)
#                 # nodes_loads[n_id] 是某一节点的负载列表
#                 for n_i_sub in nodes_loads[n_id]:
#                     if n_i_sub.belonging_vehicle_id == n_id:  # 如交付节点是车辆自身，则上传时延为0
#                         n_i_sub.up_delay = 0
#                     else:
#                         # 否则由该节点上的所有子任务等分节点的传输速率，按照上传数据大小计算时间
#                         n_i_sub.up_delay = n_i_sub.upload_size / (n_tr / nodes_loads[n_id].__len__())
#
#                 nodes_loads[n_id].sort(key=lambda x: x.upload_size)  # 节点上的所有任务按照传输时延排列，时延小的在前面
#                 tr_rank_loads = nodes_loads[n_id]
#
#                 for z in range(tr_rank_loads.__len__()):
#                     if z == 0:
#                         # 第一个到达节点的任务，总的时延为：传输时延 + 该子任务的计算时延
#                         tr_rank_loads[z].wait_delay = 0
#
#                         tr_rank_loads[z].compute_delay = tr_rank_loads[z].compute_cost / n_compute_ability
#
#                         is_reachable = calculate_two_vehicle_is_interact(
#                             math.floor(redeay_server_time + tr_rank_loads[z].up_delay +
#                                        tr_rank_loads[z].wait_delay + tr_rank_loads[z].compute_delay),
#                             tr_rank_loads[z].belonging_vehicle_id, tr_rank_loads[z].allocate_node_id)
#                         if not is_reachable:
#                             tr_rank_loads[z].punish_delay = 0.3
#                         else:
#                             tr_rank_loads[z].punish_delay = 0
#
#                     else:
#                         # 后续到达的任务，总的时延为：自身传输时延与上一个子任务的总时延中的较大者 + 该子任务的计算时延
#                         if tr_rank_loads[z - 1].up_delay + tr_rank_loads[z - 1].compute_delay > tr_rank_loads[z].up_delay:
#                             tr_rank_loads[z].wait_delay = tr_rank_loads[z - 1].up_delay + \
#                                                           tr_rank_loads[z - 1].compute_delay - tr_rank_loads[z].up_delay
#                         else:
#                             tr_rank_loads[z].wait_delay = 0
#
#                         tr_rank_loads[z].compute_delay = tr_rank_loads[z].compute_cost / n_compute_ability
#
#                         is_reachable = calculate_two_vehicle_is_interact(
#                             math.floor(redeay_server_time + tr_rank_loads[z].up_delay +
#                                        tr_rank_loads[z].wait_delay + tr_rank_loads[z].compute_delay),
#                             tr_rank_loads[z].belonging_vehicle_id, tr_rank_loads[z].allocate_node_id)
#                         if not is_reachable:
#                             tr_rank_loads[z].punish_delay = 0.3
#                         else:
#                             tr_rank_loads[z].punish_delay = 0
#
#                     tr_rank_loads[z].whole_delay = tr_rank_loads[z].up_delay + tr_rank_loads[z].wait_delay + \
#                                                    tr_rank_loads[z].compute_delay + tr_rank_loads[z].punish_delay + \
#                                                    tr_rank_loads[z].dispatch_wait_delay
#
#                     if tr_rank_loads[z].whole_delay < 0:
#                         print(tr_rank_loads[z].get_sub_task_info())
#                         print(tr_rank_loads[z].whole_delay)
#
#             else:
#                 # 云节点的处理
#                 n_compute_delay, n_tr, _ = get_info_by_node_id(n_id)
#
#                 for n_i_sub in nodes_loads[n_id]:
#                     n_i_sub.up_delay = n_i_sub.upload_size / (n_tr / nodes_loads[n_id].__len__())
#                 nodes_loads[n_id].sort(key=lambda x: x.upload_size)
#                 tr_rank_loads = nodes_loads[n_id]
#
#                 for z in range(tr_rank_loads.__len__()):
#                     tr_rank_loads[z].wait_delay = 0
#                     tr_rank_loads[z].compute_delay = 0
#                     tr_rank_loads[z].punish_delay = 0
#                     tr_rank_loads[z].whole_delay = tr_rank_loads[z].up_delay + tr_rank_loads[z].wait_delay + \
#                                                    tr_rank_loads[z].compute_delay + tr_rank_loads[z].punish_delay \
#                                                    + tr_rank_loads[z].dispatch_wait_delay
#
#     one_link = chromosomes[chromosome_id]
#     task_set = []  # 存放了每一辆车的任务 [[车的总任务], [  [车上的第一个任务],[[车上的第二个任务包含的子任务列表]], ]
#     for v_id in range(0, 99):
#         # 一辆车上的所有子任务
#         sub_tasks_in_v = [one_link[z] for z in range(one_link.__len__()) if one_link[z].belonging_vehicle_id == v_id]
#         tasks_in_v = []  # 存放一辆车的所有任务
#         # 按 任务id 聚合
#         for task_id in range(task_num_range[0], task_num_range[1]):
#             one_task_in_v = []
#             for v_s_e in sub_tasks_in_v:
#                 if v_s_e.belonging_task_order == task_id:
#                     one_task_in_v.append(v_s_e)
#             if len(one_task_in_v):
#                 tasks_in_v.append(one_task_in_v)
#
#         task_set.append(tasks_in_v)
#
#     a_link_delays = []
#     task_num_in_a_link = 0
#     # z 是一个车的所有任务
#     for z in task_set:
#         # zz 是一个车的某一个任务
#         for zz in z:
#             single_task_delay = max([zz[x].whole_delay for x in range(zz.__len__())])
#             a_link_delays.append(single_task_delay)
#             task_num_in_a_link += 1
#     # print("计算时延时的信息：----------->")
#     # print("平均任务时延：")
#     # print(sum(a_link_delays) / task_num_in_a_link)
#     # print("最大任务时延：")
#     # print(max(a_link_delays))
#
#     delay_info = {
#         # 'tasks_delay': a_link_delays,
#         'tasks_delay_sum': sum(a_link_delays),
#         'next_hop_delay': max(a_link_delays),
#         'task_number_in_a_link': task_num_in_a_link,
#         'task_average_delay': sum(a_link_delays) / task_num_in_a_link
#     }
#
#     return delay_info


# 交换两个基因的编码（交换两个子任务的节点分配）
# 计算一条染色体方案对应的时延
def compute_delay_in_a_link(redeay_server_time, chromosomes, chromosome_id):

    # 加上子任务发出时间到调度时间的差值 是为调度等待时间
    for ch in chromosomes[chromosome_id]:
        ch.dispatch_wait_delay = redeay_server_time - ch.timestamp

    # 获取所有的节点负载情况+求出了每个子任务的总时延
    nodes_loads = get_each_node_load_situation(chromosomes, chromosome_id)
    for n_id in range(nodes_loads.__len__()):
        if len(nodes_loads[n_id]):  # 只考虑有负载的节点
            if n_id != 104:
                # 非云节点，获取计算能力与传输速率
                n_compute_ability, n_tr, _ = get_info_by_node_id(n_id)
                # nodes_loads[n_id] 是某一节点的负载列表
                for n_i_sub in nodes_loads[n_id]:
                    if n_i_sub.belonging_vehicle_id == n_id:  # 如交付节点是车辆自身，则上传时延为0
                        n_i_sub.up_delay = 0
                    else:
                        # 否则由该节点上的所有子任务等分节点的传输速率，按照上传数据大小计算时间
                        n_i_sub.up_delay = n_i_sub.upload_size / (n_tr / nodes_loads[n_id].__len__())

                nodes_loads[n_id].sort(key=lambda x: x.upload_size)  # 节点上的所有任务按照传输时延排列，时延小的在前面
                tr_rank_loads = nodes_loads[n_id]

                for z in range(tr_rank_loads.__len__()):
                    if z == 0:
                        # 第一个到达节点的任务，总的时延为：传输时延 + 该子任务的计算时延
                        tr_rank_loads[z].wait_delay = 0

                        tr_rank_loads[z].compute_delay = tr_rank_loads[z].compute_cost / n_compute_ability

                        is_reachable = calculate_two_vehicle_is_interact(
                            math.floor(redeay_server_time + tr_rank_loads[z].up_delay +
                                       tr_rank_loads[z].wait_delay + tr_rank_loads[z].compute_delay),
                            tr_rank_loads[z].belonging_vehicle_id, tr_rank_loads[z].allocate_node_id)
                        if not is_reachable:
                            tr_rank_loads[z].punish_delay = 0.3
                        else:
                            tr_rank_loads[z].punish_delay = 0

                    else:
                        # 后续到达的任务，总的时延为：自身传输时延与上一个子任务的传输时延与计算时延之和中的较大者 + 该子任务的计算时延
                        if tr_rank_loads[z - 1].up_delay + tr_rank_loads[z - 1].compute_delay > tr_rank_loads[z].up_delay:
                            tr_rank_loads[z].wait_delay = tr_rank_loads[z - 1].up_delay + \
                                                          tr_rank_loads[z - 1].compute_delay - tr_rank_loads[z].up_delay
                        else:
                            tr_rank_loads[z].wait_delay = 0

                        tr_rank_loads[z].compute_delay = tr_rank_loads[z].compute_cost / n_compute_ability

                        is_reachable = calculate_two_vehicle_is_interact(
                            math.floor(redeay_server_time + tr_rank_loads[z].up_delay +
                                       tr_rank_loads[z].wait_delay + tr_rank_loads[z].compute_delay),
                            tr_rank_loads[z].belonging_vehicle_id, tr_rank_loads[z].allocate_node_id)
                        if not is_reachable:
                            tr_rank_loads[z].punish_delay = 0.3
                        else:
                            tr_rank_loads[z].punish_delay = 0

                    tr_rank_loads[z].whole_delay = tr_rank_loads[z].up_delay + tr_rank_loads[z].wait_delay + \
                                                   tr_rank_loads[z].compute_delay + tr_rank_loads[z].punish_delay +\
                                                   tr_rank_loads[z].dispatch_wait_delay

                    if tr_rank_loads[z].whole_delay < 0:
                        print(tr_rank_loads[z].get_sub_task_info())
                        print(tr_rank_loads[z].whole_delay)

            else:
                # 云节点的处理
                n_compute_delay, n_tr, _ = get_info_by_node_id(n_id)

                for n_i_sub in nodes_loads[n_id]:
                    n_i_sub.up_delay = n_i_sub.upload_size / (n_tr / nodes_loads[n_id].__len__())
                nodes_loads[n_id].sort(key=lambda x: x.upload_size)
                tr_rank_loads = nodes_loads[n_id]

                for z in range(tr_rank_loads.__len__()):
                    tr_rank_loads[z].wait_delay = 0
                    tr_rank_loads[z].compute_delay = 0
                    tr_rank_loads[z].punish_delay = 0
                    tr_rank_loads[z].whole_delay = tr_rank_loads[z].up_delay + tr_rank_loads[z].wait_delay + \
                                                   tr_rank_loads[z].compute_delay + tr_rank_loads[z].punish_delay + \
                                                   tr_rank_loads[z].dispatch_wait_delay
    # 求一个任务的时延
    one_link = chromosomes[chromosome_id]
    task_set = []  # 存放了每一辆车的任务 [[车的总任务], [  [车上的第一个任务],[[车上的第二个任务包含的子任务列表]], ]
    for v_id in range(0, 100):
        # 一辆车上的所有子任务
        sub_tasks_in_v = [one_link[z] for z in range(one_link.__len__()) if one_link[z].belonging_vehicle_id == v_id]
        tasks_in_v = []  # 存放一辆车的所有任务
        # 按 任务id 聚合
        for task_id in range(task_num_range[1]):
            one_task_in_v = []
            for v_s_e in sub_tasks_in_v:
                if v_s_e.belonging_task_order == task_id:
                    one_task_in_v.append(v_s_e)
        # 只保留了有子任务的任务
            if len(one_task_in_v):
                tasks_in_v.append(one_task_in_v)

        task_set.append(tasks_in_v)

    a_link_delays = []
    task_num_in_a_link = 0
    # z 是一个车的所有任务
    for z in task_set:
        # zz 是一个车的某一个任务
        for zz in z:
            single_task_delay = max([zz[x].whole_delay for x in range(zz.__len__())])
            a_link_delays.append(single_task_delay)
            task_num_in_a_link += 1
    print(a_link_delays)
    # print("计算时延时的信息：----------->")
    # print("平均任务时延：")
    # print(sum(a_link_delays) / task_num_in_a_link)
    # print("最大任务时延：")
    # print(max(a_link_delays))

    delay_info = {
        # 'tasks_delay': a_link_delays,
        'tasks_delay_sum': sum(a_link_delays),
        'next_hop_delay': max(a_link_delays),
        'task_number_in_a_link': task_num_in_a_link,
        'task_average_delay': sum(a_link_delays) / task_num_in_a_link
    }

    return delay_info

def exchange_allocated_node_id(gene_1, gene_2):
    ex_temp = gene_1.allocate_node_id
    gene_1.allocate_node_id = gene_2.allocate_node_id
    gene_2.allocate_node_id = ex_temp


# # 在一条染色体上，每个车都随机选择2个子任务，按一定概率交换编码
# def reverse_2_point_by_vehicle_in_a_link(link):
#     for v_id in range(0, 99):
#         sub_tasks_in_v = [link[z] for z in range(link.__len__()) if link[z].belonging_vehicle_id == v_id]
#
#         if len(sub_tasks_in_v) < 2:
#             continue
#
#         if len(sub_tasks_in_v):
#             is_reverse = random.random()
#             if is_reverse > p_reverse:
#                 s1, s2 = random.sample(sub_tasks_in_v, 2)
#                 exchange_allocated_node_id(s1, s2)
#
#
# # 按一定概率进行变异
# def link_mutation_after_cross(link):
#     for gene_x in link:
#         is_mutation = random.random()
#         if is_mutation <= p_mutation:
#             now_allocated = gene_x.allocate_node_id
#             usable_nodes = gene_x.selection_space
#
#             while True:
#                 new_allocate = random.choice(usable_nodes)
#                 if new_allocate != now_allocated:
#                     gene_x.allocate_node_id = new_allocate
#                     break


# # 按条件选择2条较好的染色体(时延比较低的)，进行交叉、变异和修复生成
# def sample_and_cross_covariance_generating(chromosomes, chromosomes_delays):
#     r_list = []
#     k_smallest = 2
#     for x in range(10):
#         r_list.append(random.randint(0, 99))
#     selected_chromosomes_delays = [chromosomes_delays[ad_i] for ad_i in r_list]
#     indices = np.argpartition(selected_chromosomes_delays, k_smallest)[:k_smallest]
#
#     chromosome_1 = copy.deepcopy(chromosomes[indices[0]])
#     chromosome_2 = copy.deepcopy(chromosomes[indices[1]])
#
#     # print(len(chromosome_1), len(chromosome_2))
#
#     n_cross = random.random()
#     if n_cross <= p_cross:
#         cut_a = random.randint(0, len(chromosome_1)-1)
#         cut_b = random.randint(0, len(chromosome_1)-1)
#         # print('cut_a:'+str(cut_a)+'cut_b:'+str(cut_b))
#         for cross_idx in range(min(cut_a, cut_b), max(cut_a, cut_b) + 1):
#             # print("cross_idx:"+str(cross_idx))
#             exchange_allocated_node_id(chromosome_1[cross_idx], chromosome_2[cross_idx])
#     else:
#         reverse_2_point_by_vehicle_in_a_link(chromosome_1)
#         reverse_2_point_by_vehicle_in_a_link(chromosome_2)
#
#     link_mutation_after_cross(chromosome_1)
#     link_mutation_after_cross(chromosome_2)
#
#     _, _, c1_bad_subs = get_node_over_distribution([chromosome_1], 0)
#     repair_overload_by_one_link([chromosome_1], 0, c1_bad_subs)
#
#     _, _, c2_bad_subs = get_node_over_distribution([chromosome_2], 0)
#     repair_overload_by_one_link([chromosome_2], 0, c2_bad_subs)

    # return chromosome_1, chromosome_2


def init_chromosomes_by_greedy(all_genes):
    greedy_chromosome = []
    orgin_mem = get_node_memory()
    for gene in all_genes:
        #
        gene_space = gene.selection_space
        # gene_space_v = [x for x in gene_space if 0 <= x <= 99]
        # gene_space_f = [x for x in gene_space if 100 <= x <= 103]
        # complete_flag = False

        # if complete_flag is False:
        #     for n_id in gene_space_v:
        #         if orgin_mem[n_id] >= gene.storage_cost:
        #             gene.allocate_node_id = n_id
        #             orgin_mem[n_id] -= gene.storage_cost
        #             complete_flag = True
        #             break

        # if complete_flag is False:
        #     for n_id in gene_space_f:
        #         if orgin_mem[n_id] >= gene.storage_cost:
        #             gene.allocate_node_id = n_id
        #             orgin_mem[n_id] -= gene.storage_cost
        #             complete_flag = True
        #             break

        # if complete_flag is False:
        gene.allocate_node_id = cloud_id

        greedy_chromosome.append(copy.deepcopy(gene))
    # print(greedy_chromosome)
    return greedy_chromosome


def initialization(dispatch_pre_time, dispatch_time):

    unsure_subtasks = []
    sure_subtasks = []
    sure_vehicles_id = []
    unsure_tasks = []
    sure_tasks = []

    # 先检测调度之前的任务请求情况
    for tx in range(dispatch_pre_time, dispatch_time+1):

        tx_total_vehicle_num_in_area = data.loc[tx].shape[0]    # tx 时间下区域内的总车辆数目

        tx_all_vehicles_sub_objects = []  # 在tx时间下，保存所有车辆发出的子任务
        tx_all_vehicles_task_objects = []

        tx_task_num = 0

        tx_vehicles_id = []

        for i in range(tx_total_vehicle_num_in_area):
            # 当前车辆的信息
            current_vehicle_id = data.loc[tx].iloc[i, 0]
            tx_vehicles_id.append(current_vehicle_id)
            try:
                fi = 'request_data/time' + str(tx) + '-vehicle' + str(current_vehicle_id) + 'request.data'
                with open(fi, 'rb') as f:
                    i_task_group = pickle.load(f)
                    # print(tx, i_task_group.vehicle_id, i_task_group.task_num)
                    tx_task_num += i_task_group.task_num

                if i_task_group.task_num > 0:
                    tx_all_vehicles_task_objects.extend(i_task_group.task_pool)
                    tx_all_vehicles_sub_objects.extend(i_task_group.get_sub_task_objects())
            except FileNotFoundError:
                continue

        if tx != dispatch_time:
            unsure_tasks.extend(tx_all_vehicles_task_objects)
            unsure_subtasks.extend(tx_all_vehicles_sub_objects)
        else:
            sure_tasks.extend(tx_all_vehicles_task_objects)
            sure_subtasks.extend(tx_all_vehicles_sub_objects)
            sure_vehicles_id = tx_vehicles_id

    # print(unsure_task_num, sure_task_num)

    for e in unsure_subtasks:
        if e.belonging_vehicle_id in sure_vehicles_id:
            # 过去的请求将在dispatch时刻被调度，需要更新其通讯空间
            vs = data.loc[dispatch_time]
            vs = vs.set_index('vehicle_id')
            dx = vs.loc[e.belonging_vehicle_id][0]
            dy = vs.loc[e.belonging_vehicle_id][1]
            e.selection_space = compute_distance_and_save_usable_nodes(dx, dy, dispatch_time)

            sure_subtasks.append(e)
            unsure_subtasks.remove(e)
    for ff in unsure_tasks:
        if ff.belonging_vehicle_id in sure_vehicles_id:
            sure_tasks.append(ff)
            unsure_tasks.remove(ff)
    global total_request_task_num, severed_task_num

    total_request_task_num += len(unsure_tasks) + len(sure_tasks)
    severed_task_num += len(sure_tasks)
    print("总共的任务数和被服务的任务数为：" + str(total_request_task_num) + " " +str(severed_task_num))
    # print(unsure_task_num, sure_task_num)
    print('时间'+str(dispatch_pre_time)+'~'+str(dispatch_time)+'内产生的总子任务数目为：' +
          str(len(unsure_subtasks) + len(sure_subtasks))+', 其中在'+str(dispatch_time)+'时刻可调度的子任务数目为' +
          str(len(sure_subtasks)) + ',不可调度子任务数为：'+str(len(unsure_subtasks)))

    # print('时间' + str(dispatch_time) + ' 下的任务数：' + str(sure_task_num))
    print('时间' + str(dispatch_time) + ' 下的基因数：' + str(sure_subtasks.__len__()))

    if len(sure_subtasks) != 0:
        # 根据当前时间段的初始状态生成100条染色体(未修复，未计算时延)
        initial_seeds = init_chromosomes_by_greedy(sure_subtasks)

        # print(initial_seeds)

        # 修复100条初代染色体，并计算时延
        initial_genetic_delays = []
        initial_hop_delays = []

        # delay_info_dict 保存了染色体的一些时延信息，可按需求读取
        delay_info_dict = compute_delay_in_a_link(dispatch_time, [initial_seeds], 0)  # 计算时延信息
        initial_genetic_delays.append(delay_info_dict['task_average_delay'])
        initial_hop_delays.append(delay_info_dict['next_hop_delay'])
    else:
        initial_seeds = []
        initial_genetic_delays = [0]
        initial_hop_delays = [1]

    return initial_seeds, initial_genetic_delays, initial_hop_delays


def save_excellent_gene_list(time_s, selected_chromosome):
    fo = 'result/time'+str(time_s)+'-best_chromosome.data'
    with open(fo, 'wb+') as f:
        pickle.dump(obj=selected_chromosome, file=f)

def severd_ratio():
    global total_request_task_num, severed_task_num
    severd_ratio = severed_task_num/total_request_task_num
    return severd_ratio