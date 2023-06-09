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
    v_trans_rate_range, v_memory_range, cloud_compute_power, cloud_transmission_rate, cloud_memory, cloud_id, \
    task_num_range, p_cross, p_mutation, p_reverse, data, v2f_limit_distance, \
    v2v_limit_distance, fogs_location,  sub_task_num_range, upload_size_range, lam, \
    download_size_range, compute_cost_range, storage_cost_range, total_tasks_number, tasks_pool_path, tasks_table_path, vehicles_attr, up_len


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
        info = [cloud_compute_power, cloud_transmission_rate, cloud_memory]
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


# # 给定一系列节点的承载状态，筛选出：任务过载的节点编号，过载量，导致节点过载的子任务(待修复的子任务)
# def judge_nodes_overload(nodes_seq):
#     overload_ids = []
#     overload_values = []
#     overload_sub_tasks = []
#     for nid in range(nodes_seq.__len__()):
#         node_loads = nodes_seq[nid]
#         node_memory = get_info_by_node_id(nid)[2]
#         cost_memory = 0
#         for sub_e in node_loads:
#             cost_memory += sub_e.storage_cost
#         if cost_memory > node_memory:
#             overload_ids.append(nid)
#             overload_sub_tasks.append(sub_e)
#             overload_values.append(cost_memory - node_memory)
#         # print(nid, node_memory, cost_memory)
#     # id 0:104（node）
#     return overload_ids, overload_values, overload_sub_tasks


# # 检查给定染色体中的过载情况，返回：染色体上过载节点集合，过载量， 待修复的子任务集合
# def get_node_over_distribution(chromosomes, chromosome_id):
#     nodes_receive_in_a_link = get_each_node_load_situation(chromosomes, chromosome_id)
#     overload_nodes_set, nodes_over_values, prepare_repair_sub_objs = judge_nodes_overload(nodes_receive_in_a_link)
#     return overload_nodes_set, nodes_over_values, prepare_repair_sub_objs

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
            n_compute_ability, n_tr, _ = get_info_by_node_id(n_id)
            # nodes_loads[n_id] 是某一节点的负载列表
            num = 0
            pre_max_up_delay = 0
            pre_up_delay = []
            local_process = [n_i_sub for n_i_sub in nodes_loads[n_id] if n_i_sub.belonging_vehicle_id == n_id]
            upload_cycle = (nodes_loads[n_id].__len__() - local_process.__len__()) / up_len + 1
            for n_i_sub in range(nodes_loads[n_id].__len__()):
                if pre_up_delay.__len__() == up_len:
                    pre_max_up_delay = max(pre_up_delay)
                    pre_up_delay = []
                if nodes_loads[n_id][n_i_sub].belonging_vehicle_id == n_id:  # 如交付节点是车辆自身，则上传时延为0
                    nodes_loads[n_id][n_i_sub].up_delay = 0
                    num += 1
                else:
                    # 否则由该节点上的所有子任务等分节点的传输速率，按照上传数据大小计算时间
                    if (n_i_sub + 1 - num)/up_len < upload_cycle:
                        nodes_loads[n_id][n_i_sub].up_delay = nodes_loads[n_id][n_i_sub].upload_size / (n_tr / up_len) + pre_max_up_delay
                        pre_up_delay.append(nodes_loads[n_id][n_i_sub].up_delay)
                    else:
                        nodes_loads[n_id][n_i_sub].up_delay = nodes_loads[n_id][n_i_sub].upload_size / (n_tr / ((nodes_loads[n_id][n_id].__len__()
                                                                                                     - local_process.__len__()) % up_len)) + pre_max_up_delay

                    # n_i_sub.up_delay = n_i_sub.upload_size / (n_tr / nodes_loads[n_id].__len__())

            nodes_loads[n_id].sort(key=lambda x: x.up_delay)  # 节点上的所有任务按照传输时延排列，时延小的在前面
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


def init_chromosomes_by_greedy(all_genes):
    greedy_chromosome = []
    orgin_mem = get_node_memory()
    # print(orgin_mem)
    for gene in all_genes:
        gene_space = gene.selection_space
        gene_space_v = [x for x in gene_space if 0 <= x <= 99]
        # print('gene_space_v' + str(gene_space_v))
        gene_space_f = [x for x in gene_space if 100 <= x <= 103]
        # print('gene_space_f' + str(gene_space_f))
        complete_flag = False

        if complete_flag is False:
            for n_id in gene_space_v:
                if orgin_mem[n_id] >= gene.storage_cost:
                    gene.allocate_node_id = n_id
                    # print("allocate_id = " + str(n_id))
                    orgin_mem[n_id] -= gene.storage_cost
                    complete_flag = True
                    break

        if complete_flag is False:
            for n_id in gene_space_f:
                if orgin_mem[n_id] >= gene.storage_cost:
                    gene.allocate_node_id = n_id
                    # print("allocate_id = " + str(n_id))
                    orgin_mem[n_id] -= gene.storage_cost
                    complete_flag = True
                    break

        if complete_flag is False:
            gene.allocate_node_id = cloud_id
            # print("allocate_id = " + str(cloud_id))

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